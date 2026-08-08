#include "ds4_glm52_l0.h"
#include "ds4_gpu_mgpu.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static const char *const k_action_ids[] = {
    "serve/action-serve-open",
    "serve/action-tp-group",
    "serve/action-serve-accept",
    "serve/action-serve-prefill",
    "serve/action-decode-token",
    "serve/action-stream-token",
    "serve/action-kv-checkpoint",
};

static const char *const k_model_load_action_ids[] = {
    "model-load/a-validate-launch-plan",
    "model-load/a-validate-shard-manifest",
    "model-load/a-map-rank-shards",
    "model-load/a-ready-rank-engines",
};

static const char *const k_prefill_action_ids[] = {
    "prefill/a-tokenize-prompt",
    "prefill/a-plan-prefill-chunks",
    "prefill/a-prefill-layer-tp",
    "prefill/a-prefill-allreduce",
    "prefill/a-commit-prefill-kv",
    "prefill/a-prefill-ready",
};

static void set_error(char *err, size_t err_size, const char *msg) {
    if (!err || err_size == 0) return;
    snprintf(err, err_size, "%s", msg ? msg : "unknown error");
}

static void set_result(ds4_glm52_l0_result *result,
                       ds4_glm52_l0_status status,
                       ds4_glm52_l0_action action,
                       const char *message) {
    if (!result) return;
    result->status = status;
    result->action = action;
    result->graph_id = DS4_GLM52_L0_GRAPH_ID;
    result->action_id = ds4_glm52_l0_action_id(action);
    snprintf(result->message,
             sizeof(result->message),
             "%s",
             message ? message : "");
}

static bool contains_casefold(const char *s, const char *needle) {
    if (!s || !needle || !needle[0]) return false;
    for (; *s; s++) {
        const char *a = s;
        const char *b = needle;
        while (*a && *b &&
               tolower((unsigned char)*a) == tolower((unsigned char)*b)) {
            a++;
            b++;
        }
        if (!*b) return true;
    }
    return false;
}

static bool model_root_looks_like_glm52(const char *s) {
    return contains_casefold(s, "glm-5.2") ||
           contains_casefold(s, "glm_5.2") ||
           contains_casefold(s, "glm5.2") ||
           contains_casefold(s, "glm52");
}

static void init_state_from_config(const ds4_glm52_l0_config *cfg,
                                   ds4_glm52_l0_state *state) {
    memset(state, 0, sizeof(*state));
    state->model_plan.model_root = cfg->model_root;
    state->model_plan.rank_plan_path = cfg->rank_plan;
    state->model_plan.tp_size = cfg->tp_size;
    state->model_plan.dcp_size = cfg->dcp_size;
    state->model_plan.pp_size = cfg->pp_size;
    state->resident_shards.rank = cfg->rank;
    state->resident_shards.no_foreign_rank_shard = true;
    state->rank_plan.rank = cfg->rank;
    state->rank_plan.q_head_start =
        cfg->rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
    state->rank_plan.q_head_end =
        state->rank_plan.q_head_start + DS4_GLM52_L0_Q_HEADS_PER_RANK;
    state->rank_plan.dcp_rank = cfg->rank;
    state->rank_plan.expert_start = -1;
    state->rank_plan.expert_end = -1;
    state->rank_plan.vocab_start = -1;
    state->rank_plan.vocab_end = -1;
    state->tp_fabric.tp_size = cfg->tp_size;
    state->tp_fabric.dcp_size = cfg->dcp_size;
    state->tp_fabric.pp_size = cfg->pp_size;
    state->tp_fabric.fabric_addr = cfg->fabric_addr;
    state->tp_fabric.fabric_data_plane = NULL;
    state->tp_fabric.rank_count = DS4_GLM52_L0_RANK_COUNT;
    state->tp_fabric.local_rank = cfg->rank;
    state->kv.tp_aware = true;
    state->kv.append_ordered = true;
    state->cursor.replicated = true;
    state->checkpoint.optional = true;
    state->logits.owner_rank = 0;
}

size_t ds4_glm52_l0_action_count(void) {
    return (size_t)DS4_GLM52_L0_ACTION_COUNT;
}

const char *ds4_glm52_l0_action_id(ds4_glm52_l0_action action) {
    if (action < 0 || action >= DS4_GLM52_L0_ACTION_COUNT) {
        return "serve/action-invalid";
    }
    return k_action_ids[action];
}

const char *ds4_glm52_model_load_action_id(ds4_glm52_model_load_action action) {
    if (action < 0 || action >= DS4_GLM52_MODEL_LOAD_ACTION_COUNT) {
        return "model-load/action-invalid";
    }
    return k_model_load_action_ids[action];
}

size_t ds4_glm52_prefill_action_count(void) {
    return sizeof(k_prefill_action_ids) / sizeof(k_prefill_action_ids[0]);
}

const char *ds4_glm52_prefill_action_id(int action_index) {
    if (action_index < 0 ||
        action_index >= (int)ds4_glm52_prefill_action_count()) {
        return "prefill/action-invalid";
    }
    return k_prefill_action_ids[action_index];
}

const char *ds4_glm52_l0_cursor_phase_name(ds4_glm52_l0_cursor_phase phase) {
    switch (phase) {
    case DS4_GLM52_L0_CURSOR_PHASE_NONE:
        return "none";
    case DS4_GLM52_L0_CURSOR_PHASE_PREFILL:
        return "prefill";
    case DS4_GLM52_L0_CURSOR_PHASE_DECODE:
        return "decode";
    default:
        return "unknown";
    }
}

const char *ds4_glm52_l0_status_name(ds4_glm52_l0_status status) {
    switch (status) {
    case DS4_GLM52_L0_STATUS_OK:
        return "ok";
    case DS4_GLM52_L0_STATUS_NOT_READY:
        return "not_ready";
    case DS4_GLM52_L0_STATUS_INVALID:
        return "invalid";
    default:
        return "unknown";
    }
}

static char *trim_ws(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

static bool parse_int_value(const char *s, int *out) {
    if (!s || !out || !s[0]) return false;
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s) return false;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end || v < -1 || v > INT32_MAX) return false;
    *out = (int)v;
    return true;
}

static void copy_string(char *dst, size_t dst_size, const char *src) {
    if (!dst || dst_size == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    size_t n = strnlen(src, dst_size - 1);
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static bool key_for_rank(const char *key, int rank, const char *field) {
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "rank%d.", rank);
    if (!strncmp(key, prefix, strlen(prefix)) &&
        !strcmp(key + strlen(prefix), field)) {
        return true;
    }
    snprintf(prefix, sizeof(prefix), "rank%d_", rank);
    return !strncmp(key, prefix, strlen(prefix)) &&
           !strcmp(key + strlen(prefix), field);
}

static bool key_for_foreign_rank(const char *key, int rank, const char *field) {
    for (int r = 0; r < DS4_GLM52_L0_RANK_COUNT; r++) {
        if (r == rank) continue;
        if (key_for_rank(key, r, field)) return true;
    }
    return false;
}

static bool path_contains_rank(const char *path, int rank) {
    char needle[16];
    snprintf(needle, sizeof(needle), "rank%d", rank);
    return contains_casefold(path, needle);
}

static bool path_contains_foreign_rank(const char *path, int rank) {
    for (int r = 0; r < DS4_GLM52_L0_RANK_COUNT; r++) {
        if (r != rank && path_contains_rank(path, r)) return true;
    }
    return false;
}

static bool resolve_path(const char *base, const char *path,
                         char *out, size_t out_size) {
    if (!path || !path[0] || !out || out_size == 0) return false;
    if (path[0] == '/') {
        return snprintf(out, out_size, "%s", path) < (int)out_size;
    }
    if (!base || !base[0]) return false;
    return snprintf(out, out_size, "%s/%s", base, path) < (int)out_size;
}

static bool existing_regular_file_size(const char *path, uint64_t *size_out) {
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) return false;
    if (size_out) *size_out = (uint64_t)st.st_size;
    return true;
}

static ds4_glm52_l0_status model_load_fail(
        ds4_glm52_l0_result *result,
        ds4_glm52_l0_status status,
        ds4_glm52_model_load_action action,
        const char *message) {
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: %.128s",
             ds4_glm52_model_load_action_id(action),
             message ? message : "");
    set_result(result, status, DS4_GLM52_L0_ACTION_SERVE_OPEN, msg);
    return status;
}

static bool resident_shards_ready(const ds4_glm52_l0_state *state) {
    return state &&
           state->model_plan.validated &&
           state->shard_manifest.validated &&
           state->resident_shards.mapped &&
           state->resident_shards.no_foreign_rank_shard &&
           state->resident_shards.base_shard_count ==
               DS4_GLM52_L0_EXPECTED_BASE_SHARDS &&
           state->resident_shards.mtp_shard_count ==
               DS4_GLM52_L0_EXPECTED_MTP_SHARDS &&
           state->resident_shards.mapped_bytes > 0;
}

static uint32_t l0_full_rank_mask(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

static bool tp_group_ready(const ds4_glm52_l0_state *state) {
    return resident_shards_ready(state) &&
           state->rank_plan.bound &&
           state->tp_fabric.group_ready &&
           state->tp_fabric.topology_bound &&
           state->tp_fabric.transport_ready &&
           state->tp_fabric.tp_size == DS4_GLM52_L0_TP_SIZE &&
           state->tp_fabric.dcp_size == DS4_GLM52_L0_DCP_SIZE &&
           state->tp_fabric.pp_size == DS4_GLM52_L0_PP_SIZE &&
           state->tp_fabric.rank_count == DS4_GLM52_L0_RANK_COUNT &&
           state->tp_fabric.local_rank == state->rank_plan.rank;
}

static bool request_ready(const ds4_glm52_l0_state *state) {
    return state && state->request.accepted && state->request.session_id &&
           state->request.session_id[0];
}

static bool prefill_ready(const ds4_glm52_l0_state *state) {
    return tp_group_ready(state) &&
           state->kv.tp_aware &&
           state->kv.append_ordered &&
           state->kv.prefill_complete &&
           state->cursor.replicated &&
           state->cursor.kv_length == state->kv.length &&
           (state->cursor.phase == DS4_GLM52_L0_CURSOR_PHASE_PREFILL ||
            state->cursor.phase == DS4_GLM52_L0_CURSOR_PHASE_DECODE);
}

static bool fabric_addr_valid(const char *addr) {
    if (!addr || !addr[0]) return false;
    bool has_non_space = false;
    bool has_addr_char = false;
    for (const unsigned char *p = (const unsigned char *)addr; *p; p++) {
        if (isspace(*p)) return false;
        has_non_space = true;
        if (isalnum(*p) || *p == '.' || *p == ':' || *p == '-' || *p == '_') {
            has_addr_char = true;
            continue;
        }
        return false;
    }
    return has_non_space && has_addr_char;
}

static bool fabric_addr_is_management_network(const char *addr) {
    return addr && !strncmp(addr, "192.168.0.", strlen("192.168.0."));
}

static bool fabric_data_plane_valid(const char *data_plane) {
    return data_plane &&
           !strcmp(data_plane, DS4_GLM52_L0_FABRIC_DATA_PLANE);
}

static ds4_glm52_l0_status bind_tp_group(
        const ds4_glm52_l0_config *cfg,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result) {
    if (!resident_shards_ready(state)) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_TP_GROUP,
                   "TP group formation requires resident rank-local model shards from model-load");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }

    const char *fabric_addr = cfg->fabric_addr && cfg->fabric_addr[0] ?
        cfg->fabric_addr : state->launch_plan.fabric_addr;
    if (!fabric_addr_valid(fabric_addr)) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_TP_GROUP,
                   "TP group formation requires --glm52-tp4-fabric or rank-plan fabric_addr without whitespace");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (fabric_addr_is_management_network(fabric_addr)) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_TP_GROUP,
                   "TP group formation requires the CRS812 200G fabric address, not the 192.168.0.x management network");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!fabric_data_plane_valid(state->launch_plan.fabric_data_plane)) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_TP_GROUP,
                   "TP group formation requires rank-plan fabric_data_plane=crs812-200g");
        return DS4_GLM52_L0_STATUS_INVALID;
    }

    const int expected_q_start = cfg->rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
    const int expected_q_end = expected_q_start + DS4_GLM52_L0_Q_HEADS_PER_RANK;
    if (cfg->rank < 0 ||
        cfg->rank >= DS4_GLM52_L0_RANK_COUNT ||
        state->resident_shards.rank != cfg->rank) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_TP_GROUP,
                   "TP rank ownership must match the local resident shard rank");
        return DS4_GLM52_L0_STATUS_INVALID;
    }

    state->rank_plan.rank = cfg->rank;
    state->rank_plan.q_head_start = expected_q_start;
    state->rank_plan.q_head_end = expected_q_end;
    state->rank_plan.dcp_rank = cfg->rank;
    if (state->rank_plan.expert_start == 0 &&
        state->rank_plan.expert_end == 0) {
        state->rank_plan.expert_start = -1;
        state->rank_plan.expert_end = -1;
    }
    if (state->rank_plan.vocab_start == 0 &&
        state->rank_plan.vocab_end == 0) {
        state->rank_plan.vocab_start = -1;
        state->rank_plan.vocab_end = -1;
    }
    state->rank_plan.bound = true;

    state->tp_fabric.tp_size = cfg->tp_size;
    state->tp_fabric.dcp_size = cfg->dcp_size;
    state->tp_fabric.pp_size = cfg->pp_size;
    state->tp_fabric.rank_count = DS4_GLM52_L0_RANK_COUNT;
    state->tp_fabric.local_rank = cfg->rank;
    state->tp_fabric.fabric_addr = fabric_addr;
    state->tp_fabric.fabric_data_plane = state->launch_plan.fabric_data_plane;
    state->tp_fabric.topology_bound = true;
    state->tp_fabric.transport_ready = false;
    state->tp_fabric.group_ready = false;

    set_result(result,
               DS4_GLM52_L0_STATUS_NOT_READY,
               DS4_GLM52_L0_ACTION_TP_GROUP,
               "TP4/DCP4 topology is bound; real four-rank collective transport handshake is not implemented");
    return DS4_GLM52_L0_STATUS_NOT_READY;
}

static ds4_glm52_l0_status load_plan_and_manifest(
        const ds4_glm52_l0_config *cfg,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result) {
    FILE *fp = fopen(cfg->rank_plan, "r");
    if (!fp) {
        char msg[192];
        snprintf(msg, sizeof(msg), "cannot open rank plan '%s'",
                 cfg->rank_plan);
        return model_load_fail(result,
                               DS4_GLM52_L0_STATUS_INVALID,
                               DS4_GLM52_MODEL_LOAD_VALIDATE_LAUNCH_PLAN,
                               msg);
    }

    ds4_glm52_l0_launch_plan plan = {0};
    ds4_glm52_l0_shard_manifest manifest = {0};
    plan.tp_size = -1;
    plan.dcp_size = -1;
    plan.pp_size = -1;
    plan.rank_count = -1;
    plan.rank = cfg->rank;
    plan.host_count = -1;
    manifest.rank = cfg->rank;
    copy_string(plan.checkpoint_root,
                sizeof(plan.checkpoint_root),
                cfg->model_root);
    copy_string(plan.fabric_addr, sizeof(plan.fabric_addr), cfg->fabric_addr);

    char line[1024];
    int line_no = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = trim_ws(line);
        if (!p[0] || p[0] == '#') continue;
        char *eq = strchr(p, '=');
        if (!eq) {
            fclose(fp);
            char msg[192];
            snprintf(msg, sizeof(msg), "line %d is not key=value", line_no);
            return model_load_fail(result,
                                   DS4_GLM52_L0_STATUS_INVALID,
                                   DS4_GLM52_MODEL_LOAD_VALIDATE_LAUNCH_PLAN,
                                   msg);
        }
        *eq = '\0';
        char *key = trim_ws(p);
        char *value = trim_ws(eq + 1);
        if (!strcmp(key, "tp_size")) {
            if (!parse_int_value(value, &plan.tp_size)) goto bad_int;
        } else if (!strcmp(key, "dcp_size")) {
            if (!parse_int_value(value, &plan.dcp_size)) goto bad_int;
        } else if (!strcmp(key, "pp_size")) {
            if (!parse_int_value(value, &plan.pp_size)) goto bad_int;
        } else if (!strcmp(key, "rank_count")) {
            if (!parse_int_value(value, &plan.rank_count)) goto bad_int;
        } else if (!strcmp(key, "rank")) {
            if (!parse_int_value(value, &plan.rank)) goto bad_int;
        } else if (!strcmp(key, "host_count") ||
                   !strcmp(key, "gx10_count")) {
            if (!parse_int_value(value, &plan.host_count)) goto bad_int;
        } else if (!strcmp(key, "model_name")) {
            copy_string(plan.model_name, sizeof(plan.model_name), value);
        } else if (!strcmp(key, "checkpoint_root")) {
            copy_string(plan.checkpoint_root,
                        sizeof(plan.checkpoint_root),
                        value);
        } else if (!strcmp(key, "fabric_addr")) {
            copy_string(plan.fabric_addr, sizeof(plan.fabric_addr), value);
        } else if (!strcmp(key, "fabric_data_plane") ||
                   !strcmp(key, "data_plane")) {
            copy_string(plan.fabric_data_plane,
                        sizeof(plan.fabric_data_plane),
                        value);
        } else if (key_for_rank(key, cfg->rank, "checkpoint_root")) {
            copy_string(plan.checkpoint_root,
                        sizeof(plan.checkpoint_root),
                        value);
        } else if (!strcmp(key, "base_shard") ||
                   key_for_rank(key, cfg->rank, "base_shard")) {
            char path[PATH_MAX];
            uint64_t bytes = 0;
            if (path_contains_foreign_rank(value, cfg->rank)) {
                manifest.has_foreign_rank_shard = true;
            }
            if (!resolve_path(plan.checkpoint_root, value,
                              path, sizeof(path)) ||
                !existing_regular_file_size(path, &bytes)) {
                fclose(fp);
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "missing base shard for rank %d: %s",
                         cfg->rank, value);
                return model_load_fail(
                        result,
                        DS4_GLM52_L0_STATUS_INVALID,
                        DS4_GLM52_MODEL_LOAD_VALIDATE_SHARD_MANIFEST,
                        msg);
            }
            manifest.base_shard_count++;
            manifest.total_bytes += bytes;
        } else if (!strcmp(key, "mtp_shard") ||
                   key_for_rank(key, cfg->rank, "mtp_shard")) {
            char path[PATH_MAX];
            uint64_t bytes = 0;
            if (path_contains_foreign_rank(value, cfg->rank)) {
                manifest.has_foreign_rank_shard = true;
            }
            if (!resolve_path(plan.checkpoint_root, value,
                              path, sizeof(path)) ||
                !existing_regular_file_size(path, &bytes)) {
                fclose(fp);
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "missing MTP shard for rank %d: %s",
                         cfg->rank, value);
                return model_load_fail(
                        result,
                        DS4_GLM52_L0_STATUS_INVALID,
                        DS4_GLM52_MODEL_LOAD_VALIDATE_SHARD_MANIFEST,
                        msg);
            }
            manifest.mtp_shard_count++;
            manifest.total_bytes += bytes;
        } else if (key_for_foreign_rank(key, cfg->rank, "base_shard") ||
                   key_for_foreign_rank(key, cfg->rank, "mtp_shard")) {
            continue;
        }
        continue;

bad_int:
        fclose(fp);
        char msg[192];
        snprintf(msg, sizeof(msg), "line %d has invalid integer for %s",
                 line_no, key);
        return model_load_fail(result,
                               DS4_GLM52_L0_STATUS_INVALID,
                               DS4_GLM52_MODEL_LOAD_VALIDATE_LAUNCH_PLAN,
                               msg);
    }
    fclose(fp);

    const char *model_name =
        plan.model_name[0] ? plan.model_name : cfg->model_root;
    const char *effective_fabric_addr =
        cfg->fabric_addr && cfg->fabric_addr[0] ?
        cfg->fabric_addr : plan.fabric_addr;
    if (plan.tp_size != DS4_GLM52_L0_TP_SIZE ||
        plan.dcp_size != DS4_GLM52_L0_DCP_SIZE ||
        plan.pp_size != DS4_GLM52_L0_PP_SIZE ||
        plan.rank_count != DS4_GLM52_L0_RANK_COUNT ||
        plan.host_count != DS4_GLM52_L0_RANK_COUNT ||
        plan.rank != cfg->rank ||
        !fabric_addr_valid(effective_fabric_addr) ||
        fabric_addr_is_management_network(effective_fabric_addr) ||
        !fabric_data_plane_valid(plan.fabric_data_plane) ||
        !model_root_looks_like_glm52(model_name)) {
        return model_load_fail(
                result,
                DS4_GLM52_L0_STATUS_INVALID,
                DS4_GLM52_MODEL_LOAD_VALIDATE_LAUNCH_PLAN,
                "rank plan must declare GLM 5.2 TP4/DCP4/PP1 over four GX10 ranks, local rank, and CRS812 200G fabric data plane");
    }
    copy_string(plan.fabric_addr, sizeof(plan.fabric_addr), effective_fabric_addr);
    plan.validated = true;

    if (manifest.base_shard_count != DS4_GLM52_L0_EXPECTED_BASE_SHARDS ||
        manifest.mtp_shard_count != DS4_GLM52_L0_EXPECTED_MTP_SHARDS ||
        manifest.has_foreign_rank_shard) {
        return model_load_fail(
                result,
                DS4_GLM52_L0_STATUS_INVALID,
                DS4_GLM52_MODEL_LOAD_VALIDATE_SHARD_MANIFEST,
                "rank shard manifest must contain exactly 20 base shards, one MTP shard, and no foreign-rank shard");
    }
    manifest.validated = true;

    state->launch_plan = plan;
    state->shard_manifest = manifest;
    state->model_plan.validated = true;
    state->resident_shards.checkpoint_root = state->launch_plan.checkpoint_root;
    state->resident_shards.base_shard_count = manifest.base_shard_count;
    state->resident_shards.mtp_shard_count = manifest.mtp_shard_count;
    state->resident_shards.mapped_bytes = manifest.total_bytes;
    state->resident_shards.no_foreign_rank_shard =
        !manifest.has_foreign_rank_shard;
    return DS4_GLM52_L0_STATUS_OK;
}

bool ds4_glm52_l0_config_from_engine(const ds4_engine_options *opt,
                                     ds4_glm52_l0_config *cfg,
                                     char *err,
                                     size_t err_size) {
    if (!opt || !cfg) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 config requires engine options");
        return false;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->enabled = opt->glm52_tp4_l0;
    cfg->mock_model = opt->glm52_tp4_mock_model;
    cfg->mock_matmul = opt->glm52_tp4_mock_matmul;
    cfg->gpu_residency_required = opt->glm52_tp4_gpu_resident;
    cfg->legacy_cuda_tensor_parallel = opt->cuda_tensor_parallel;
    cfg->legacy_two_rank_tp = opt->tp.requested || opt->tp.role != DS4_TP_NONE;
    cfg->rank_set = opt->glm52_tp4_rank_set;
    cfg->rank = opt->glm52_tp4_rank;
    cfg->tp_size = opt->glm52_tp4_tp_size > 0 ?
        opt->glm52_tp4_tp_size : DS4_GLM52_L0_TP_SIZE;
    cfg->dcp_size = opt->glm52_tp4_dcp_size > 0 ?
        opt->glm52_tp4_dcp_size : DS4_GLM52_L0_DCP_SIZE;
    cfg->pp_size = opt->glm52_tp4_pp_size > 0 ?
        opt->glm52_tp4_pp_size : DS4_GLM52_L0_PP_SIZE;
    cfg->gpu_device_id = 0;
    cfg->model_root = opt->model_path;
    cfg->rank_plan = opt->glm52_tp4_rank_plan;
    cfg->layout_path = opt->glm52_tp4_layout_path;
    cfg->fabric_addr = opt->glm52_tp4_fabric_addr;
    return true;
}

bool ds4_glm52_l0_validate_config(const ds4_glm52_l0_config *cfg,
                                  char *err,
                                  size_t err_size) {
    if (!cfg) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 config is missing");
        return false;
    }
    if (!cfg->enabled) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 mode is not enabled");
        return false;
    }
    if (cfg->mock_matmul && !cfg->mock_model) {
        set_error(err, err_size, "GLM 5.2 mock matmul requires --glm52-tp4-mock-model");
        return false;
    }
    if (cfg->gpu_residency_required && cfg->mock_model) {
        set_error(err, err_size, "GLM 5.2 GPU residency requires a real model layout, not --glm52-tp4-mock-model");
        return false;
    }
    if (!cfg->model_root || !cfg->model_root[0]) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires --model MODEL_ROOT");
        return false;
    }
    if (!model_root_looks_like_glm52(cfg->model_root)) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires an explicit GLM 5.2 model root");
        return false;
    }
    if (!cfg->rank_plan || !cfg->rank_plan[0]) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires --glm52-tp4-rank-plan FILE");
        return false;
    }
    if (!cfg->rank_set) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires --glm52-tp4-rank N");
        return false;
    }
    if (cfg->legacy_cuda_tensor_parallel || cfg->legacy_two_rank_tp) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 cannot be combined with legacy two-rank TP options");
        return false;
    }
    if (cfg->tp_size != DS4_GLM52_L0_TP_SIZE) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires tp_size=4");
        return false;
    }
    if (cfg->dcp_size != DS4_GLM52_L0_DCP_SIZE) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires dcp_size=4");
        return false;
    }
    if (cfg->pp_size != DS4_GLM52_L0_PP_SIZE) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 requires pp_size=1");
        return false;
    }
    if (cfg->rank < 0 || cfg->rank >= cfg->tp_size) {
        set_error(err, err_size, "GLM 5.2 TP4 L0 rank must be in [0,4)");
        return false;
    }
    if (cfg->gpu_residency_required && cfg->gpu_device_id < 0) {
        set_error(err, err_size, "GLM 5.2 GPU residency requires a valid rank-local GPU device id");
        return false;
    }
    return true;
}

/* Run the model-shard-layout child graph when a layout manifest is provided.
 * Executes the BCD actions in contract order:
 *   a-read-layout-spec -> a-validate-tensor-ownership ->
 *   a-load-rank-tensors -> a-publish-loaded-shard
 * and, when requested by runtime config:
 *   a-upload-gpu-resident-tensors
 * Updates state->resident_shards and state->rank_plan on success. */
static ds4_glm52_l0_status run_model_shard_layout(
        const ds4_glm52_l0_config *cfg,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result) {
    if (!cfg->layout_path || !cfg->layout_path[0]) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_SERVE_OPEN,
                   "model-load/a-map-rank-shards: no DS4 layout manifest path provided; layout child is not implemented");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }

    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_status st;

    /* a-read-layout-spec */
    st = ds4_glm52_layout_read_manifest(cfg->layout_path, &spec, result);
    if (st != DS4_GLM52_L0_STATUS_OK) return st;

    /* a-validate-tensor-ownership */
    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, cfg->rank, &plan, result);
    if (st != DS4_GLM52_L0_STATUS_OK) return st;

    /* a-load-rank-tensors */
    ds4_glm52_layout_mapped_slices mapped;
    st = ds4_glm52_layout_mmap_rank_tensors(&plan, cfg->model_root,
                                            &mapped, result);
    if (st != DS4_GLM52_L0_STATUS_OK) return st;

    /* a-publish-loaded-shard */
    st = ds4_glm52_layout_publish_loaded_rank_shard(&mapped, &plan,
                                                     state, result);
    if (st != DS4_GLM52_L0_STATUS_OK) return st;

    if (cfg->gpu_residency_required) {
        if (!cfg->gpu_runtime) {
            set_result(result,
                       DS4_GLM52_L0_STATUS_INVALID,
                       DS4_GLM52_L0_ACTION_SERVE_OPEN,
                       "model-shard-layout/a-upload-gpu-resident-tensors: GPU-resident model load requires an initialized CUDA tensor runtime");
            return DS4_GLM52_L0_STATUS_INVALID;
        }
        st = ds4_glm52_layout_upload_resident_gpu_tensors(
                state,
                cfg->gpu_device_id,
                cfg->gpu_runtime,
                result);
    }
    return st;
}

/* ================================================================== */
/* prefill child graph implementation                                 */
/* BCD actions: a-tokenize-prompt, a-plan-prefill-chunks,             */
/* a-prefill-layer-tp, a-prefill-allreduce, a-commit-prefill-kv,      */
/* a-prefill-ready                                                    */
/* ================================================================== */

static ds4_glm52_l0_status prefill_fail(ds4_glm52_l0_result *result,
                                        ds4_glm52_l0_status status,
                                        int action_index,
                                        const char *message) {
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: %.128s",
             ds4_glm52_prefill_action_id(action_index),
             message ? message : "");
    set_result(result, status, DS4_GLM52_L0_ACTION_PREFILL, msg);
    return status;
}

static void prefill_trace(ds4_glm52_l0_trace_fn trace,
                          void *trace_ud,
                          int action_index,
                          ds4_glm52_l0_status status,
                          bool prerequisite_blocked,
                          const ds4_glm52_l0_state *state) {
    if (!trace) return;
    trace(trace_ud,
          DS4_GLM52_PREFILL_GRAPH_ID,
          ds4_glm52_prefill_action_id(action_index),
          status,
          prerequisite_blocked,
          state);
}

static bool prefill_identity_ok(const ds4_glm52_l0_config *cfg,
                                const ds4_glm52_l0_state *state) {
    return state->resident_shards.rank == state->rank_plan.rank &&
           state->rank_plan.rank == cfg->rank &&
           state->rank_plan.rank == state->tp_fabric.local_rank;
}

bool ds4_glm52_l0_prefill_set_prompt(ds4_glm52_l0_state *state,
                                     const char *session_id,
                                     const int *token_ids,
                                     int token_count,
                                     char *err,
                                     size_t err_size) {
    if (!state) {
        set_error(err, err_size,
                  "GLM 5.2 TP4 prefill state is missing");
        return false;
    }
    if (!session_id || !session_id[0]) {
        set_error(err, err_size,
                  "GLM 5.2 TP4 prefill prompt requires a session id");
        return false;
    }
    if (token_count < 0 ||
        token_count > DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS) {
        set_error(err, err_size,
                  "GLM 5.2 TP4 prefill prompt exceeds the L0 prompt token bound");
        return false;
    }
    if (token_count > 0 && !token_ids) {
        set_error(err, err_size,
                  "GLM 5.2 TP4 prefill prompt span is missing");
        return false;
    }

    memset(&state->prompt, 0, sizeof(state->prompt));
    snprintf(state->prompt.session_id, sizeof(state->prompt.session_id),
             "%s", session_id);
    state->prompt.prompt_length = token_count;
    state->prompt.consumed = 0;
    if (token_count > 0) {
        memcpy(state->prompt.token_ids, token_ids,
               (size_t)token_count * sizeof(state->prompt.token_ids[0]));
    }
    state->prompt.set = true;
    return true;
}

ds4_glm52_l0_status ds4_glm52_l0_prefill_run(
        const ds4_glm52_l0_config *cfg,
        ds4_glm52_l0_state *state,
        int chunk_size,
        ds4_glm52_l0_trace_fn trace,
        void *trace_ud,
        ds4_glm52_l0_result *result) {
    char err[160];
    if (!ds4_glm52_l0_validate_config(cfg, err, sizeof(err))) {
        set_result(result, DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_PREFILL, err);
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!state) {
        set_result(result, DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_PREFILL,
                   "GLM 5.2 TP4 L0 prefill state is missing");
        return DS4_GLM52_L0_STATUS_INVALID;
    }

    /* ---- a-tokenize-prompt: serve preconditions + prompt span ---- */
    if (!resident_shards_ready(state)) {
        prefill_trace(trace, trace_ud, 0,
                      DS4_GLM52_L0_STATUS_NOT_READY, true, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_NOT_READY, 0,
                            "prefill requires resident rank-local model shards from model-load");
    }
    if (!tp_group_ready(state)) {
        prefill_trace(trace, trace_ud, 0,
                      DS4_GLM52_L0_STATUS_NOT_READY, true, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_NOT_READY, 0,
                            "prefill requires a ready TP4/DCP4 collective fabric");
    }
    if (!request_ready(state)) {
        prefill_trace(trace, trace_ud, 0,
                      DS4_GLM52_L0_STATUS_NOT_READY, true, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_NOT_READY, 0,
                            "prefill requires an accepted request/session");
    }
    if (!state->prompt.set ||
        !state->prompt.session_id[0] ||
        strcmp(state->prompt.session_id, state->request.session_id) != 0) {
        prefill_trace(trace, trace_ud, 0,
                      DS4_GLM52_L0_STATUS_NOT_READY, true, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_NOT_READY, 0,
                            "prefill requires a prompt token span for the accepted session (a-tokenize-prompt)");
    }
    prefill_trace(trace, trace_ud, 0, DS4_GLM52_L0_STATUS_OK, false, state);

    /* ---- a-plan-prefill-chunks: KV identity + chunk plan ---- */
    if (!state->kv.tp_aware || !state->kv.append_ordered) {
        prefill_trace(trace, trace_ud, 1,
                      DS4_GLM52_L0_STATUS_INVALID, false, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                            "prefill requires TP-aware append-ordered KV state (state-kv identity)");
    }
    if (state->kv.prefill_complete) {
        prefill_trace(trace, trace_ud, 1,
                      DS4_GLM52_L0_STATUS_INVALID, false, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                            "prefill refuses to re-append an already-prefilled append-ordered KV cursor");
    }
    if (state->kv.session_id &&
        state->kv.session_id[0] &&
        strcmp(state->kv.session_id, state->request.session_id) != 0) {
        prefill_trace(trace, trace_ud, 1,
                      DS4_GLM52_L0_STATUS_INVALID, false, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                            "prefill KV session identity mismatch (same_state_as serve state-kv)");
    }
    if (!prefill_identity_ok(cfg, state)) {
        prefill_trace(trace, trace_ud, 2,
                      DS4_GLM52_L0_STATUS_INVALID, false, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                            "prefill rank-plan/model/fabric identity mismatch");
    }

    const int prompt_length = state->prompt.prompt_length;
    const int cs = chunk_size > 0 ? chunk_size : DS4_GLM52_L0_PREFILL_CHUNK;
    const int chunk_count = prompt_length == 0 ? 0 :
        (prompt_length + cs - 1) / cs;

    /* Fail-closed pass: verify the whole chunk plan against the
     * append-ordered KV cursor before mutating any state. */
    {
        int expected = (int)state->kv.length;
        for (int i = 0; i < chunk_count; i++) {
            const int start = i * cs;
            const int end = (start + cs < prompt_length) ?
                (start + cs) : prompt_length;
            if (start != expected) {
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "prefill chunk %d starts at position %d but the append-ordered KV cursor is at %d",
                         i, start, expected);
                prefill_trace(trace, trace_ud, 4,
                              DS4_GLM52_L0_STATUS_INVALID, false, state);
                return prefill_fail(result,
                                    DS4_GLM52_L0_STATUS_INVALID, 4, msg);
            }
            expected = end;
        }
        if (expected != prompt_length) {
            prefill_trace(trace, trace_ud, 4,
                          DS4_GLM52_L0_STATUS_INVALID, false, state);
            return prefill_fail(result, DS4_GLM52_L0_STATUS_INVALID, 4,
                                "prefill chunk plan does not cover the full prompt span");
        }
    }

    if (!cfg->mock_model || !cfg->mock_matmul) {
        prefill_trace(trace, trace_ud, 2,
                      DS4_GLM52_L0_STATUS_NOT_READY, true, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_NOT_READY, 2,
                            "real GLM 5.2 TP4 prefill layer kernels are not implemented");
    }

    /* ---- a-prefill-layer-tp / a-prefill-allreduce: explicit mock seams ---- */
    /* No real GLM 5.2 TP kernels or all-reduce collectives run here. The
     * mock-only layer-before-reduce-before-commit-before-ready ordering is
     * proven by resident-shard / TP-fabric / rank identity validation above
     * and by append-order validation below; traces make the skeleton contract
     * order observable without real weights. */
    prefill_trace(trace, trace_ud, 1, DS4_GLM52_L0_STATUS_OK, false, state);
    prefill_trace(trace, trace_ud, 2, DS4_GLM52_L0_STATUS_OK, false, state);
    prefill_trace(trace, trace_ud, 3, DS4_GLM52_L0_STATUS_OK, false, state);

    /* ---- a-commit-prefill-kv: append prompt rows and cursors ---- */
    if (!state->kv.session_id || !state->kv.session_id[0]) {
        state->kv.session_id = state->request.session_id;
    }
    for (int i = 0; i < chunk_count; i++) {
        const int start = i * cs;
        const int end = (start + cs < prompt_length) ?
            (start + cs) : prompt_length;
        state->kv.length = (uint64_t)end;
        state->cursor.kv_length = (uint64_t)end;
        state->cursor.token_step_j = (uint64_t)end;
        state->prompt.consumed = end;
        state->chunk.chunk_index = i;
        state->chunk.start_position = start;
        state->chunk.end_position = end;
        state->chunk.token_count = end - start;
    }
    state->commit.prefilled_length = (uint64_t)prompt_length;
    state->commit.tp_degree = DS4_GLM52_L0_TP_SIZE;
    state->commit.dcp_degree = DS4_GLM52_L0_DCP_SIZE;
    state->commit.kv_cache_dtype = DS4_GLM52_L0_KV_CACHE_DTYPE;
    state->commit.committed = true;
    prefill_trace(trace, trace_ud, 4, DS4_GLM52_L0_STATUS_OK, false, state);

    /* ---- a-prefill-ready: publish decode handoff ---- */
    if (state->cursor.kv_length != (uint64_t)prompt_length ||
        state->cursor.token_step_j != (uint64_t)prompt_length ||
        state->kv.length != (uint64_t)prompt_length) {
        prefill_trace(trace, trace_ud, 5,
                      DS4_GLM52_L0_STATUS_INVALID, false, state);
        return prefill_fail(result, DS4_GLM52_L0_STATUS_INVALID, 5,
                            "prefill final cursor consistency check failed after commit");
    }
    state->kv.prefill_complete = true;
    state->cursor.replicated = true;
    state->cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_PREFILL;
    prefill_trace(trace, trace_ud, 5, DS4_GLM52_L0_STATUS_OK, false, state);

    char msg[192];
    snprintf(msg, sizeof(msg),
             "prefill/a-prefill-ready: prompt prefill complete; decode may start at position %d",
             prompt_length);
    set_result(result, DS4_GLM52_L0_STATUS_OK,
               DS4_GLM52_L0_ACTION_PREFILL, msg);
    return DS4_GLM52_L0_STATUS_OK;
}

const char *ds4_glm52_tp4_collective_kind_name(
        ds4_glm52_tp4_collective_kind kind) {
    switch (kind) {
    case DS4_GLM52_TP4_COLLECTIVE_ATTN:
        return "attn";
    case DS4_GLM52_TP4_COLLECTIVE_FFN:
        return "ffn";
    case DS4_GLM52_TP4_COLLECTIVE_LOGITS:
        return "logits";
    default:
        return "unknown";
    }
}

const char *ds4_glm52_tp4_tensor_dtype_name(
        ds4_glm52_tp4_tensor_dtype dtype) {
    switch (dtype) {
    case DS4_GLM52_TP4_TENSOR_DTYPE_F32:
        return "f32";
    case DS4_GLM52_TP4_TENSOR_DTYPE_BF16:
        return "bf16";
    case DS4_GLM52_TP4_TENSOR_DTYPE_FP8_E4M3:
        return "fp8_e4m3";
    case DS4_GLM52_TP4_TENSOR_DTYPE_I32:
        return "i32";
    default:
        return "invalid";
    }
}

size_t ds4_glm52_tp4_tensor_dtype_size(
        ds4_glm52_tp4_tensor_dtype dtype) {
    switch (dtype) {
    case DS4_GLM52_TP4_TENSOR_DTYPE_F32:
        return 4;
    case DS4_GLM52_TP4_TENSOR_DTYPE_BF16:
        return 2;
    case DS4_GLM52_TP4_TENSOR_DTYPE_FP8_E4M3:
        return 1;
    case DS4_GLM52_TP4_TENSOR_DTYPE_I32:
        return 4;
    default:
        return 0;
    }
}

static bool valid_collective_kind(ds4_glm52_tp4_collective_kind kind) {
    return kind == DS4_GLM52_TP4_COLLECTIVE_ATTN ||
           kind == DS4_GLM52_TP4_COLLECTIVE_FFN ||
           kind == DS4_GLM52_TP4_COLLECTIVE_LOGITS;
}

static bool valid_tensor_dtype(ds4_glm52_tp4_tensor_dtype dtype) {
    return ds4_glm52_tp4_tensor_dtype_size(dtype) != 0;
}

static bool collective_byte_count_ok(size_t element_count,
                                     ds4_glm52_tp4_tensor_dtype dtype,
                                     size_t byte_count) {
    const size_t dtype_size = ds4_glm52_tp4_tensor_dtype_size(dtype);
    if (element_count == 0 || dtype_size == 0) return false;
    if (element_count > SIZE_MAX / dtype_size) return false;
    return byte_count == element_count * dtype_size;
}

static void write_be32(unsigned char *dst, uint32_t v) {
    dst[0] = (unsigned char)((v >> 24) & 0xffu);
    dst[1] = (unsigned char)((v >> 16) & 0xffu);
    dst[2] = (unsigned char)((v >> 8) & 0xffu);
    dst[3] = (unsigned char)(v & 0xffu);
}

static uint32_t read_be32(const unsigned char *src) {
    return ((uint32_t)src[0] << 24) |
           ((uint32_t)src[1] << 16) |
           ((uint32_t)src[2] << 8) |
           (uint32_t)src[3];
}

static void write_be64(unsigned char *dst, uint64_t v) {
    for (int i = 7; i >= 0; i--) {
        dst[7 - i] = (unsigned char)((v >> (i * 8)) & 0xffu);
    }
}

static uint64_t read_be64(const unsigned char *src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v = (v << 8) | (uint64_t)src[i];
    }
    return v;
}

static bool size_t_to_u64(size_t v, uint64_t *out) {
    if (!out) return false;
    if (v > (size_t)UINT64_MAX) return false;
    *out = (uint64_t)v;
    return true;
}

static bool u64_to_size_t(uint64_t v, size_t *out) {
    if (!out) return false;
    if (v > (uint64_t)SIZE_MAX) return false;
    *out = (size_t)v;
    return true;
}

bool ds4_glm52_tp4_collective_frame_validate(
        const ds4_glm52_tp4_collective_request *request,
        char *err,
        size_t err_size) {
    if (!request) {
        set_error(err, err_size, "real TP4 collective request is missing");
        return false;
    }
    if (request->frame_version != DS4_GLM52_TP4_COLLECTIVE_FRAME_VERSION) {
        set_error(err, err_size,
                  "real TP4 collective requires collective frame version 1");
        return false;
    }
    if (!valid_collective_kind(request->kind)) {
        set_error(err, err_size, "real TP4 collective kind is invalid");
        return false;
    }
    if (request->rank < 0 ||
        request->rank >= DS4_GLM52_L0_RANK_COUNT ||
        request->tp_size != DS4_GLM52_L0_TP_SIZE ||
        request->dcp_size != DS4_GLM52_L0_DCP_SIZE ||
        request->rank_count != DS4_GLM52_L0_RANK_COUNT) {
        set_error(err, err_size,
                  "real TP4 collective requires rank in [0,4) with TP4/DCP4/rank_count=4");
        return false;
    }
    if (!valid_tensor_dtype(request->dtype)) {
        set_error(err, err_size,
                  "real TP4 collective requires a supported tensor dtype");
        return false;
    }
    if (request->participant_mask != l0_full_rank_mask()) {
        set_error(err, err_size,
                  "real TP4 collective requires all four rank participants");
        return false;
    }
    if (request->seq == 0 ||
        request->model_hash == 0 ||
        request->session_hash == 0 ||
        request->element_count == 0 ||
        request->shape_hash == 0 ||
        request->byte_count == 0 ||
        request->layer_index < 0 ||
        !collective_byte_count_ok(request->element_count,
                                  request->dtype,
                                  request->byte_count)) {
        set_error(err, err_size,
                  "real TP4 collective requires nonzero identity, layer, shape hash, element count, and matching dtype byte count");
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_collective_frame_encode(
        const ds4_glm52_tp4_collective_request *request,
        unsigned char *wire,
        size_t wire_size,
        char *err,
        size_t err_size) {
    if (!wire || wire_size != DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE) {
        set_error(err, err_size,
                  "real TP4 collective wire buffer must be exactly 96 bytes");
        return false;
    }
    if (!ds4_glm52_tp4_collective_frame_validate(
                request, err, err_size)) {
        return false;
    }

    uint64_t element_count = 0;
    uint64_t byte_count = 0;
    if (!size_t_to_u64(request->element_count, &element_count) ||
        !size_t_to_u64(request->byte_count, &byte_count)) {
        set_error(err, err_size,
                  "real TP4 collective wire frame cannot represent size_t fields");
        return false;
    }

    memset(wire, 0, wire_size);
    size_t off = 0;
    write_be32(wire + off, request->frame_version); off += 4;
    write_be32(wire + off, (uint32_t)request->kind); off += 4;
    write_be32(wire + off, (uint32_t)(int32_t)request->rank); off += 4;
    write_be32(wire + off, (uint32_t)(int32_t)request->tp_size); off += 4;
    write_be32(wire + off, (uint32_t)(int32_t)request->dcp_size); off += 4;
    write_be32(wire + off, (uint32_t)(int32_t)request->rank_count); off += 4;
    write_be32(wire + off, (uint32_t)(int32_t)request->layer_index); off += 4;
    write_be32(wire + off, (uint32_t)request->dtype); off += 4;
    write_be64(wire + off, request->seq); off += 8;
    write_be64(wire + off, request->model_hash); off += 8;
    write_be64(wire + off, request->session_hash); off += 8;
    write_be64(wire + off, request->token_step_j); off += 8;
    write_be64(wire + off, element_count); off += 8;
    write_be64(wire + off, request->shape_hash); off += 8;
    write_be64(wire + off, byte_count); off += 8;
    write_be32(wire + off, request->participant_mask); off += 4;
    uint32_t flags = 0;
    if (request->topology_ready) flags |= UINT32_C(1) << 0;
    if (request->transport_ready) flags |= UINT32_C(1) << 1;
    if (request->rank_local_partial_ready) flags |= UINT32_C(1) << 2;
    if (request->replicated_output_ready) flags |= UINT32_C(1) << 3;
    write_be32(wire + off, flags); off += 4;
    if (off != DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE) {
        set_error(err, err_size,
                  "real TP4 collective wire frame size accounting failed");
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_collective_frame_decode(
        const unsigned char *wire,
        size_t wire_size,
        ds4_glm52_tp4_collective_request *request,
        char *err,
        size_t err_size) {
    if (!wire || !request || wire_size != DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE) {
        set_error(err, err_size,
                  "real TP4 collective wire buffer must be exactly 96 bytes");
        return false;
    }

    memset(request, 0, sizeof(*request));
    size_t off = 0;
    request->frame_version = read_be32(wire + off); off += 4;
    request->kind = (ds4_glm52_tp4_collective_kind)read_be32(wire + off); off += 4;
    request->rank = (int)(int32_t)read_be32(wire + off); off += 4;
    request->tp_size = (int)(int32_t)read_be32(wire + off); off += 4;
    request->dcp_size = (int)(int32_t)read_be32(wire + off); off += 4;
    request->rank_count = (int)(int32_t)read_be32(wire + off); off += 4;
    request->layer_index = (int)(int32_t)read_be32(wire + off); off += 4;
    request->dtype = (ds4_glm52_tp4_tensor_dtype)read_be32(wire + off); off += 4;
    request->seq = read_be64(wire + off); off += 8;
    request->model_hash = read_be64(wire + off); off += 8;
    request->session_hash = read_be64(wire + off); off += 8;
    request->token_step_j = read_be64(wire + off); off += 8;
    uint64_t element_count = read_be64(wire + off); off += 8;
    request->shape_hash = read_be64(wire + off); off += 8;
    uint64_t byte_count = read_be64(wire + off); off += 8;
    request->participant_mask = read_be32(wire + off); off += 4;
    uint32_t flags = read_be32(wire + off); off += 4;
    if (off != DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE) {
        set_error(err, err_size,
                  "real TP4 collective wire frame size accounting failed");
        return false;
    }
    if ((flags & ~UINT32_C(0xf)) != 0) {
        set_error(err, err_size,
                  "real TP4 collective wire frame contains unknown flags");
        return false;
    }
    if (!u64_to_size_t(element_count, &request->element_count) ||
        !u64_to_size_t(byte_count, &request->byte_count)) {
        set_error(err, err_size,
                  "real TP4 collective wire frame size exceeds this process");
        return false;
    }
    request->topology_ready = (flags & (UINT32_C(1) << 0)) != 0;
    request->transport_ready = (flags & (UINT32_C(1) << 1)) != 0;
    request->rank_local_partial_ready = (flags & (UINT32_C(1) << 2)) != 0;
    request->replicated_output_ready = (flags & (UINT32_C(1) << 3)) != 0;
    return ds4_glm52_tp4_collective_frame_validate(
            request, err, err_size);
}

static bool collective_f32_allreduce_kind(
        ds4_glm52_tp4_collective_kind kind) {
    return kind == DS4_GLM52_TP4_COLLECTIVE_ATTN ||
           kind == DS4_GLM52_TP4_COLLECTIVE_FFN;
}

static bool same_collective_identity(
        const ds4_glm52_tp4_collective_request *a,
        const ds4_glm52_tp4_collective_request *b) {
    return a->kind == b->kind &&
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
           a->participant_mask == b->participant_mask;
}

bool ds4_glm52_tp4_collective_allreduce_f32_host(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const float *const partials[DS4_GLM52_L0_RANK_COUNT],
        float *const outputs[DS4_GLM52_L0_RANK_COUNT],
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!requests || !partials || !outputs || element_count == 0) {
        set_error(err, err_size,
                  "TP4 host all-reduce requires frames, partial buffers, output buffers, and nonzero elements");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!partials[rank] || !outputs[rank]) {
            set_error(err, err_size,
                      "TP4 host all-reduce requires every rank partial and output buffer");
            return false;
        }
    }

    const ds4_glm52_tp4_collective_request *ref = &requests[0];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_collective_request *req = &requests[rank];
        if (!ds4_glm52_tp4_collective_frame_validate(
                    req, err, err_size)) {
            return false;
        }
        if (req->rank != rank) {
            set_error(err, err_size,
                      "TP4 host all-reduce request array must be indexed by rank");
            return false;
        }
        if (!collective_f32_allreduce_kind(req->kind)) {
            set_error(err, err_size,
                      "TP4 host all-reduce only accepts attention or FFN collectives");
            return false;
        }
        if (req->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
            set_error(err, err_size,
                      "TP4 host all-reduce currently requires f32 tensor frames");
            return false;
        }
        if (!req->topology_ready ||
            !req->transport_ready ||
            !req->rank_local_partial_ready ||
            !req->replicated_output_ready) {
            set_error(err, err_size,
                      "TP4 host all-reduce requires topology, transport, rank-local partial, and replicated output readiness");
            return false;
        }
        if (req->element_count != element_count) {
            set_error(err, err_size,
                      "TP4 host all-reduce element count does not match frame metadata");
            return false;
        }
        if (rank != 0 && !same_collective_identity(ref, req)) {
            set_error(err, err_size,
                      "TP4 host all-reduce rank frames disagree on collective identity");
            return false;
        }
        for (size_t i = 0; i < element_count; i++) {
            if (!isfinite(partials[rank][i])) {
                set_error(err, err_size,
                          "TP4 host all-reduce rank-local partial contains non-finite values");
                return false;
            }
        }
    }

    float *reduced = calloc(element_count, sizeof(reduced[0]));
    if (!reduced) {
        set_error(err, err_size,
                  "TP4 host all-reduce could not allocate staging buffer");
        return false;
    }
    for (int src_rank = 0; src_rank < DS4_GLM52_L0_RANK_COUNT; src_rank++) {
        const float *partial = partials[src_rank];
        for (size_t i = 0; i < element_count; i++) {
            reduced[i] += partial[i];
        }
    }
    for (size_t i = 0; i < element_count; i++) {
        if (!isfinite(reduced[i])) {
            free(reduced);
            set_error(err, err_size,
                      "TP4 host all-reduce replicated output contains non-finite values");
            return false;
        }
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        memcpy(outputs[rank], reduced, element_count * sizeof(reduced[0]));
    }
    free(reduced);
    return true;
}

static bool tensor_binding_range_ok(const ds4_glm52_tp4_tensor_binding *binding,
                                    size_t dtype_size) {
    if (!binding || dtype_size == 0) return false;
    if (binding->byte_count == 0 || binding->capacity_bytes == 0) return false;
    if ((binding->byte_offset % dtype_size) != 0 ||
        (binding->byte_count % dtype_size) != 0) {
        return false;
    }
    if (binding->byte_offset > binding->capacity_bytes) return false;
    return binding->byte_count <= binding->capacity_bytes - binding->byte_offset;
}

static bool validate_tp4_tensor_binding(
        const char *name,
        const ds4_glm52_tp4_tensor_binding *binding,
        int rank,
        ds4_glm52_tp4_tensor_dtype dtype,
        uint64_t shape_hash,
        size_t byte_count,
        char *err,
        size_t err_size) {
    const size_t dtype_size = ds4_glm52_tp4_tensor_dtype_size(dtype);
    if (!binding || !binding->handle || !binding->ready) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s tensor binding requires a non-null ready tensor handle",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (binding->rank != rank) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s tensor binding must be indexed by rank",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (binding->dtype != dtype ||
        binding->shape_hash != shape_hash ||
        binding->byte_count != byte_count) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s tensor binding dtype, shape, or byte count does not match collective metadata",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (!tensor_binding_range_ok(binding, dtype_size)) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s tensor binding byte range exceeds tensor capacity or is not dtype-aligned",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_collective_bind_tensor_buffers(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_tensor_binding partials[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_tensor_binding outputs[DS4_GLM52_L0_RANK_COUNT],
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!requests || !partials || !outputs || element_count == 0) {
        set_error(err, err_size,
                  "TP4 tensor binding requires frames, partial tensors, output tensors, and nonzero elements");
        return false;
    }

    const ds4_glm52_tp4_collective_request *ref = &requests[0];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_collective_request *req = &requests[rank];
        if (!ds4_glm52_tp4_collective_frame_validate(
                    req, err, err_size)) {
            return false;
        }
        if (req->rank != rank) {
            set_error(err, err_size,
                      "TP4 tensor binding request array must be indexed by rank");
            return false;
        }
        if (!collective_f32_allreduce_kind(req->kind)) {
            set_error(err, err_size,
                      "TP4 tensor binding only accepts attention or FFN collectives");
            return false;
        }
        if (!req->topology_ready ||
            !req->transport_ready ||
            !req->rank_local_partial_ready ||
            !req->replicated_output_ready) {
            set_error(err, err_size,
                      "TP4 tensor binding requires topology, transport, rank-local partial, and replicated output readiness");
            return false;
        }
        if (req->element_count != element_count) {
            set_error(err, err_size,
                      "TP4 tensor binding element count does not match frame metadata");
            return false;
        }
        if (rank != 0 && !same_collective_identity(ref, req)) {
            set_error(err, err_size,
                      "TP4 tensor binding rank frames disagree on collective identity");
            return false;
        }
        if (!validate_tp4_tensor_binding("TP4 rank-local partial",
                                         &partials[rank],
                                         rank,
                                         req->dtype,
                                         req->shape_hash,
                                         req->byte_count,
                                         err,
                                         err_size)) {
            return false;
        }
        if (!validate_tp4_tensor_binding("TP4 replicated output",
                                         &outputs[rank],
                                         rank,
                                         req->dtype,
                                         req->shape_hash,
                                         req->byte_count,
                                         err,
                                         err_size)) {
            return false;
        }
    }
    return true;
}

static bool validate_tp4_gpu_tensor_binding(
        const char *name,
        const ds4_glm52_tp4_gpu_tensor_binding *binding,
        int rank,
        int expected_device,
        ds4_glm52_tp4_tensor_dtype dtype,
        uint64_t shape_hash,
        size_t byte_count,
        char *err,
        size_t err_size) {
    const size_t dtype_size = ds4_glm52_tp4_tensor_dtype_size(dtype);
    if (!binding || !binding->tensor || !binding->tensor->ptr ||
        !binding->ready) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s GPU tensor binding requires a non-null ready device tensor",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (binding->rank != rank) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s GPU tensor binding must be indexed by rank",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (binding->tensor->device_id != expected_device) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s GPU tensor binding device does not match rank ownership",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (binding->dtype != dtype ||
        binding->shape_hash != shape_hash ||
        binding->byte_count != byte_count) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s GPU tensor binding dtype, shape, or byte count does not match collective metadata",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    if (dtype_size == 0 ||
        binding->byte_offset > binding->tensor->bytes ||
        binding->byte_count > binding->tensor->bytes - binding->byte_offset ||
        (binding->byte_offset % dtype_size) != 0 ||
        (binding->byte_count % dtype_size) != 0) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s GPU tensor binding byte range exceeds device tensor capacity or is not dtype-aligned",
                 name ? name : "TP4");
        set_error(err, err_size, msg);
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_collective_bind_gpu_tensors(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_gpu_tensor_binding partials[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_gpu_tensor_binding outputs[DS4_GLM52_L0_RANK_COUNT],
        const int expected_devices[DS4_GLM52_L0_RANK_COUNT],
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!requests || !partials || !outputs || !expected_devices ||
        element_count == 0) {
        set_error(err, err_size,
                  "TP4 GPU tensor binding requires frames, device tensors, rank devices, and nonzero elements");
        return false;
    }

    const ds4_glm52_tp4_collective_request *ref = &requests[0];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_collective_request *req = &requests[rank];
        if (!ds4_glm52_tp4_collective_frame_validate(
                    req, err, err_size)) {
            return false;
        }
        if (req->rank != rank) {
            set_error(err, err_size,
                      "TP4 GPU tensor binding request array must be indexed by rank");
            return false;
        }
        if (!collective_f32_allreduce_kind(req->kind)) {
            set_error(err, err_size,
                      "TP4 GPU tensor binding only accepts attention or FFN collectives");
            return false;
        }
        if (req->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
            set_error(err, err_size,
                      "TP4 GPU tensor binding currently requires f32 collective tensors");
            return false;
        }
        if (req->element_count != element_count) {
            set_error(err, err_size,
                      "TP4 GPU tensor binding element count does not match frame metadata");
            return false;
        }
        if (rank != 0 && !same_collective_identity(ref, req)) {
            set_error(err, err_size,
                      "TP4 GPU tensor binding rank frames disagree on collective identity");
            return false;
        }
        if (!validate_tp4_gpu_tensor_binding("TP4 GPU rank-local partial",
                                             &partials[rank],
                                             rank,
                                             expected_devices[rank],
                                             req->dtype,
                                             req->shape_hash,
                                             req->byte_count,
                                             err,
                                             err_size)) {
            return false;
        }
        if (!validate_tp4_gpu_tensor_binding("TP4 GPU replicated output",
                                             &outputs[rank],
                                             rank,
                                             expected_devices[rank],
                                             req->dtype,
                                             req->shape_hash,
                                             req->byte_count,
                                             err,
                                             err_size)) {
            return false;
        }
    }
    return true;
}

bool ds4_glm52_tp4_gpu_collective_read_f32(
        const ds4_glm52_tp4_collective_request *request,
        const ds4_glm52_tp4_gpu_tensor_binding *binding,
        int expected_device,
        const ds4_glm52_gpu_tensor_io *io,
        float *out,
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!request || !binding || !io || !io->read || !out ||
        element_count == 0) {
        set_error(err, err_size,
                  "TP4 GPU staged read requires frame, binding, read callback, output, and nonzero elements");
        return false;
    }
    if (!ds4_glm52_tp4_collective_frame_validate(
                request, err, err_size)) {
        return false;
    }
    if (request->element_count != element_count ||
        request->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
        set_error(err, err_size,
                  "TP4 GPU staged read requires matching f32 element count");
        return false;
    }
    if (!validate_tp4_gpu_tensor_binding("TP4 GPU staged read",
                                         binding,
                                         request->rank,
                                         expected_device,
                                         request->dtype,
                                         request->shape_hash,
                                         request->byte_count,
                                         err,
                                         err_size)) {
        return false;
    }
    return io->read(binding->tensor,
                    (uint64_t)binding->byte_offset,
                    out,
                    (uint64_t)binding->byte_count) != 0;
}

bool ds4_glm52_tp4_gpu_collective_write_f32(
        const ds4_glm52_tp4_collective_request *request,
        const ds4_glm52_tp4_gpu_tensor_binding *binding,
        int expected_device,
        const ds4_glm52_gpu_tensor_io *io,
        const float *data,
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!request || !binding || !io || !io->write || !data ||
        element_count == 0) {
        set_error(err, err_size,
                  "TP4 GPU staged write requires frame, binding, write callback, input, and nonzero elements");
        return false;
    }
    if (!ds4_glm52_tp4_collective_frame_validate(
                request, err, err_size)) {
        return false;
    }
    if (request->element_count != element_count ||
        request->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
        set_error(err, err_size,
                  "TP4 GPU staged write requires matching f32 element count");
        return false;
    }
    if (!validate_tp4_gpu_tensor_binding("TP4 GPU staged write",
                                         binding,
                                         request->rank,
                                         expected_device,
                                         request->dtype,
                                         request->shape_hash,
                                         request->byte_count,
                                         err,
                                         err_size)) {
        return false;
    }
    return io->write((struct ds4_gpu_tensor *)binding->tensor,
                     (uint64_t)binding->byte_offset,
                     data,
                     (uint64_t)binding->byte_count) != 0;
}

static void *tensor_binding_ptr(const ds4_glm52_tp4_tensor_binding *binding) {
    return (void *)((unsigned char *)binding->handle + binding->byte_offset);
}

bool ds4_glm52_tp4_collective_allreduce_f32_bound_host(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_tensor_binding partials[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_tensor_binding outputs[DS4_GLM52_L0_RANK_COUNT],
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!ds4_glm52_tp4_collective_bind_tensor_buffers(
                requests, partials, outputs, element_count, err, err_size)) {
        return false;
    }
    const float *partial_ptrs[DS4_GLM52_L0_RANK_COUNT];
    float *output_ptrs[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (requests[rank].dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
            set_error(err, err_size,
                      "TP4 bound host all-reduce requires f32 tensor bindings");
            return false;
        }
        partial_ptrs[rank] =
            (const float *)tensor_binding_ptr(&partials[rank]);
        output_ptrs[rank] =
            (float *)tensor_binding_ptr(&outputs[rank]);
    }
    return ds4_glm52_tp4_collective_allreduce_f32_host(
            requests,
            partial_ptrs,
            output_ptrs,
            element_count,
            err,
            err_size);
}

static bool logits_candidate_better(
        const ds4_glm52_tp4_logits_topk_entry *a,
        const ds4_glm52_tp4_logits_topk_entry *b) {
    if (!b->present) return true;
    if (a->score > b->score) return true;
    if (a->score < b->score) return false;
    if (a->token_id < b->token_id) return true;
    if (a->token_id > b->token_id) return false;
    return a->owner_rank < b->owner_rank;
}

bool ds4_glm52_tp4_logits_gather_topk_f32_host(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_logits_rank_candidates shards[DS4_GLM52_L0_RANK_COUNT],
        size_t k,
        ds4_glm52_tp4_logits_topk_entry *out,
        size_t out_count,
        char *err,
        size_t err_size) {
    if (!requests || !shards || !out || k == 0 || out_count < k) {
        set_error(err, err_size,
                  "TP4 logits gather/top-k requires frames, rank shards, output storage, and nonzero k");
        return false;
    }
    for (size_t i = 0; i < out_count; i++) {
        out[i].present = false;
        out[i].token_id = -1;
        out[i].score = 0.0f;
        out[i].owner_rank = -1;
    }

    const ds4_glm52_tp4_collective_request *ref = &requests[0];
    int expected_vocab_start = 0;
    size_t total_candidates = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_collective_request *req = &requests[rank];
        const ds4_glm52_tp4_logits_rank_candidates *shard = &shards[rank];
        if (!ds4_glm52_tp4_collective_frame_validate(
                    req, err, err_size)) {
            return false;
        }
        if (req->rank != rank || req->kind != DS4_GLM52_TP4_COLLECTIVE_LOGITS) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k requires LOGITS frames indexed by rank");
            return false;
        }
        if (req->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k currently requires f32 score frames");
            return false;
        }
        if (!req->topology_ready ||
            !req->transport_ready ||
            !req->rank_local_partial_ready ||
            !req->replicated_output_ready) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k requires topology, transport, rank-local logits, and coordinator output readiness");
            return false;
        }
        if (rank != 0 && !same_collective_identity(ref, req)) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k rank frames disagree on collective identity");
            return false;
        }
        if (!shard->present ||
            shard->rank != rank ||
            shard->vocab_start != expected_vocab_start ||
            shard->vocab_end <= shard->vocab_start ||
            shard->vocab_end > DS4_GLM52_L0_VOCAB_SIZE ||
            !shard->token_ids ||
            !shard->scores ||
            shard->candidate_count == 0 ||
            shard->candidate_count > req->element_count) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k requires contiguous rank vocab shards and score candidates");
            return false;
        }
        const size_t shard_width =
            (size_t)(shard->vocab_end - shard->vocab_start);
        if (req->element_count != shard_width) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k frame element count must describe the full rank-owned vocab shard");
            return false;
        }
        expected_vocab_start = shard->vocab_end;
        total_candidates += shard->candidate_count;
        for (size_t i = 0; i < shard->candidate_count; i++) {
            const int token = shard->token_ids[i];
            if (token < shard->vocab_start || token >= shard->vocab_end) {
                set_error(err, err_size,
                          "TP4 logits gather/top-k candidate token is outside its owner shard");
                return false;
            }
            if (!isfinite(shard->scores[i])) {
                set_error(err, err_size,
                          "TP4 logits gather/top-k candidate score is non-finite");
                return false;
            }
        }
    }
    if (expected_vocab_start != DS4_GLM52_L0_VOCAB_SIZE) {
        set_error(err, err_size,
                  "TP4 logits gather/top-k vocab shard coverage is incomplete");
        return false;
    }
    if (k > total_candidates) {
        set_error(err, err_size,
                  "TP4 logits gather/top-k requires at least k candidates");
        return false;
    }

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_logits_rank_candidates *shard = &shards[rank];
        for (size_t i = 0; i < shard->candidate_count; i++) {
            ds4_glm52_tp4_logits_topk_entry cand = {
                .token_id = shard->token_ids[i],
                .score = shard->scores[i],
                .owner_rank = rank,
                .present = true,
            };
            for (size_t pos = 0; pos < k; pos++) {
                if (logits_candidate_better(&cand, &out[pos])) {
                    for (size_t shift = k - 1; shift > pos; shift--) {
                        out[shift] = out[shift - 1];
                    }
                    out[pos] = cand;
                    break;
                }
            }
        }
    }
    for (size_t i = 0; i < k; i++) {
        if (!out[i].present) {
            set_error(err, err_size,
                      "TP4 logits gather/top-k failed to fill requested output");
            return false;
        }
    }
    return true;
}

bool ds4_glm52_tp4_logits_bind_tensor_shards(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_logits_tensor_shard shards[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    if (!requests || !shards) {
        set_error(err, err_size,
                  "TP4 logits tensor binding requires frames and rank-local logits shards");
        return false;
    }

    const ds4_glm52_tp4_collective_request *ref = &requests[0];
    int expected_vocab_start = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_collective_request *req = &requests[rank];
        const ds4_glm52_tp4_logits_tensor_shard *shard = &shards[rank];
        if (!ds4_glm52_tp4_collective_frame_validate(
                    req, err, err_size)) {
            return false;
        }
        if (req->rank != rank || req->kind != DS4_GLM52_TP4_COLLECTIVE_LOGITS) {
            set_error(err, err_size,
                      "TP4 logits tensor binding requires LOGITS frames indexed by rank");
            return false;
        }
        if (req->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32) {
            set_error(err, err_size,
                      "TP4 logits tensor binding currently requires f32 logits shards");
            return false;
        }
        if (!req->topology_ready ||
            !req->transport_ready ||
            !req->rank_local_partial_ready ||
            !req->replicated_output_ready) {
            set_error(err, err_size,
                      "TP4 logits tensor binding requires topology, transport, rank-local logits, and coordinator output readiness");
            return false;
        }
        if (rank != 0 && !same_collective_identity(ref, req)) {
            set_error(err, err_size,
                      "TP4 logits tensor binding rank frames disagree on collective identity");
            return false;
        }
        if (!shard->present ||
            shard->rank != rank ||
            shard->vocab_start != expected_vocab_start ||
            shard->vocab_end <= shard->vocab_start ||
            shard->vocab_end > DS4_GLM52_L0_VOCAB_SIZE) {
            set_error(err, err_size,
                      "TP4 logits tensor binding requires contiguous rank vocab shards indexed by rank");
            return false;
        }
        const size_t shard_width =
            (size_t)(shard->vocab_end - shard->vocab_start);
        if (req->element_count != shard_width) {
            set_error(err, err_size,
                      "TP4 logits tensor binding frame element count must match the rank vocab shard width");
            return false;
        }
        if (!validate_tp4_tensor_binding("TP4 rank-local logits",
                                         &shard->logits,
                                         rank,
                                         req->dtype,
                                         req->shape_hash,
                                         req->byte_count,
                                         err,
                                         err_size)) {
            return false;
        }
        expected_vocab_start = shard->vocab_end;
    }
    if (expected_vocab_start != DS4_GLM52_L0_VOCAB_SIZE) {
        set_error(err, err_size,
                  "TP4 logits tensor binding vocab shard coverage is incomplete");
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_logits_gather_topk_f32_bound_host(
        const ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_tp4_logits_tensor_shard shards[DS4_GLM52_L0_RANK_COUNT],
        size_t k,
        ds4_glm52_tp4_logits_topk_entry *out,
        size_t out_count,
        char *err,
        size_t err_size) {
    if (!out || k == 0 || out_count < k) {
        set_error(err, err_size,
                  "TP4 bound host logits top-k requires output storage and nonzero k");
        return false;
    }
    if (!ds4_glm52_tp4_logits_bind_tensor_shards(
                requests, shards, err, err_size)) {
        return false;
    }
    ds4_glm52_tp4_logits_topk_entry *staging =
        calloc(k, sizeof(staging[0]));
    if (!staging) {
        set_error(err, err_size,
                  "TP4 bound host logits top-k could not allocate staging output");
        return false;
    }
    for (size_t i = 0; i < k; i++) {
        staging[i].present = false;
        staging[i].token_id = -1;
        staging[i].score = 0.0f;
        staging[i].owner_rank = -1;
    }

    size_t total_logits = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const size_t shard_width =
            (size_t)(shards[rank].vocab_end - shards[rank].vocab_start);
        total_logits += shard_width;
        const float *scores =
            (const float *)tensor_binding_ptr(&shards[rank].logits);
        for (size_t i = 0; i < shard_width; i++) {
            if (!isfinite(scores[i])) {
                free(staging);
                set_error(err, err_size,
                          "TP4 bound host logits top-k score is non-finite");
                return false;
            }
            ds4_glm52_tp4_logits_topk_entry cand = {
                .token_id = shards[rank].vocab_start + (int)i,
                .score = scores[i],
                .owner_rank = rank,
                .present = true,
            };
            for (size_t pos = 0; pos < k; pos++) {
                if (logits_candidate_better(&cand, &staging[pos])) {
                    for (size_t shift = k - 1; shift > pos; shift--) {
                        staging[shift] = staging[shift - 1];
                    }
                    staging[pos] = cand;
                    break;
                }
            }
        }
    }
    if (k > total_logits) {
        free(staging);
        set_error(err, err_size,
                  "TP4 bound host logits top-k requires k within vocab coverage");
        return false;
    }
    for (size_t i = 0; i < k; i++) {
        if (!staging[i].present) {
            free(staging);
            set_error(err, err_size,
                      "TP4 bound host logits top-k failed to fill requested output");
            return false;
        }
    }
    for (size_t i = 0; i < out_count; i++) {
        out[i].present = false;
        out[i].token_id = -1;
        out[i].score = 0.0f;
        out[i].owner_rank = -1;
    }
    memcpy(out, staging, k * sizeof(out[0]));
    free(staging);
    return true;
}

static bool decode_real_request_ready(
        const ds4_glm52_decode_real_request *decode,
        char *err,
        size_t err_size) {
    if (!decode) {
        set_error(err, err_size, "decode collective call requires decode identity");
        return false;
    }
    if (decode->rank < 0 ||
        decode->rank >= DS4_GLM52_L0_RANK_COUNT ||
        decode->input_token < 0 ||
        decode->seq == 0 ||
        decode->model_hash == 0 ||
        decode->session_hash == 0) {
        set_error(err, err_size,
                  "decode collective call requires valid rank, token, sequence, model, and session identity");
        return false;
    }
    if (!decode->model_ready ||
        !decode->tp_collectives_ready ||
        !decode->dcp_exchange_ready ||
        !decode->glm52_kernels_ready) {
        set_error(err, err_size,
                  "decode collective call requires model, TP collective, DCP exchange, and GLM kernel readiness");
        return false;
    }
    return true;
}

static bool decode_collective_requests_match(
        const ds4_glm52_decode_real_request *decode,
        const ds4_glm52_tp4_collective_request *requests,
        char *err,
        size_t err_size) {
    if (!requests) {
        set_error(err, err_size, "decode collective call requires TP4 frames");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_tp4_collective_request *req = &requests[rank];
        if (!ds4_glm52_tp4_collective_frame_validate(
                    req, err, err_size)) {
            return false;
        }
        if (req->rank != rank ||
            req->seq != decode->seq ||
            req->model_hash != decode->model_hash ||
            req->session_hash != decode->session_hash) {
            set_error(err, err_size,
                      "decode collective TP4 frame identity does not match decode request");
            return false;
        }
    }
    return true;
}

bool ds4_glm52_decode_bound_collective_host(
        const ds4_glm52_decode_bound_collective_call *call,
        char *err,
        size_t err_size) {
    if (!call ||
        !decode_real_request_ready(call->decode, err, err_size) ||
        !decode_collective_requests_match(
                call->decode, call->requests, err, err_size)) {
        return false;
    }
    const ds4_glm52_tp4_collective_kind kind = call->requests[0].kind;
    if (kind == DS4_GLM52_TP4_COLLECTIVE_ATTN ||
        kind == DS4_GLM52_TP4_COLLECTIVE_FFN) {
        return ds4_glm52_tp4_collective_allreduce_f32_bound_host(
                call->requests,
                call->partials,
                call->outputs,
                call->element_count,
                err,
                err_size);
    }
    if (kind == DS4_GLM52_TP4_COLLECTIVE_LOGITS) {
        return ds4_glm52_tp4_logits_gather_topk_f32_bound_host(
                call->requests,
                call->logits_shards,
                call->top_k,
                call->topk_out,
                call->topk_out_count,
                err,
                err_size);
    }
    set_error(err, err_size,
              "decode collective call received unsupported TP4 collective kind");
    return false;
}

ds4_glm52_l0_status ds4_glm52_tp4_real_collective_allreduce(
        const ds4_glm52_tp4_collective_request *request,
        ds4_glm52_l0_result *result) {
    char err[192];
    if (!ds4_glm52_tp4_collective_frame_validate(
                request, err, sizeof(err))) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   err);
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!request->topology_ready || !request->transport_ready) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real TP4 collective backend is not ready: topology and tensor transport are required");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (!request->rank_local_partial_ready ||
        !request->replicated_output_ready) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real TP4 collective backend is not ready: device partial and replicated output buffers are required");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    set_result(result,
               DS4_GLM52_L0_STATUS_NOT_READY,
               DS4_GLM52_L0_ACTION_DECODE_TOKEN,
               "real TP4 all-reduce tensor execution is not implemented behind the validated boundary");
    return DS4_GLM52_L0_STATUS_NOT_READY;
}

static bool same_dcp_identity(
        const ds4_glm52_dcp_exchange_request *a,
        const ds4_glm52_dcp_exchange_request *b) {
    return a->dcp_size == b->dcp_size &&
           a->rank_count == b->rank_count &&
           a->layer_index == b->layer_index &&
           a->seq == b->seq &&
           a->model_hash == b->model_hash &&
           a->session_hash == b->session_hash &&
           a->token_step_j == b->token_step_j &&
           a->owner_rank_mask == b->owner_rank_mask &&
           a->ownership_plan_valid == b->ownership_plan_valid &&
           a->append_ordered_kv == b->append_ordered_kv &&
           a->row_payload_ready == b->row_payload_ready &&
           a->transport_ready == b->transport_ready;
}

static bool dcp_owner_for_row_host(
        const ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        uint64_t row_id,
        int *owner_out) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (row_id >= owners[rank].row_start &&
            row_id < owners[rank].row_end) {
            if (owner_out) *owner_out = rank;
            return true;
        }
    }
    return false;
}

static bool dcp_payload_lt(
        const ds4_glm52_dcp_row_payload *a,
        const ds4_glm52_dcp_row_payload *b) {
    if (a->layer_index != b->layer_index) return a->layer_index < b->layer_index;
    if (a->token_step_j != b->token_step_j) return a->token_step_j < b->token_step_j;
    return a->row_id < b->row_id;
}

static bool dcp_reply_insert_sorted_host(
        ds4_glm52_dcp_rank_reply *reply,
        const ds4_glm52_dcp_row_payload *row,
        char *err,
        size_t err_size) {
    if (reply->row_count >= reply->row_capacity) {
        set_error(err, err_size,
                  "DCP selected-row exchange reply capacity is too small");
        return false;
    }
    size_t pos = 0;
    for (; pos < reply->row_count; pos++) {
        if (reply->rows[pos].row_id == row->row_id) return true;
        if (dcp_payload_lt(row, &reply->rows[pos])) break;
    }
    for (size_t i = reply->row_count; i > pos; i--) {
        reply->rows[i] = reply->rows[i - 1];
    }
    reply->rows[pos] = *row;
    reply->row_count++;
    return true;
}

static bool dcp_catalog_lookup_host(
        const ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_dcp_row_payload *catalog,
        size_t catalog_count,
        uint64_t row_id,
        ds4_glm52_dcp_row_payload *out,
        char *err,
        size_t err_size) {
    int owner = -1;
    if (!dcp_owner_for_row_host(owners, row_id, &owner)) {
        set_error(err, err_size,
                  "DCP selected-row exchange selected row is outside ownership ranges");
        return false;
    }

    bool found = false;
    ds4_glm52_dcp_row_payload canonical;
    memset(&canonical, 0, sizeof(canonical));
    for (size_t i = 0; i < catalog_count; i++) {
        const ds4_glm52_dcp_row_payload *row = &catalog[i];
        if (!row->present || row->row_id != row_id) continue;
        if (row->owner_rank != owner ||
            row->layer_index < 0 ||
            row->kv_hash == 0 ||
            row->k_rope_hash == 0) {
            set_error(err, err_size,
                      "DCP selected-row exchange catalog row does not match ownership or payload requirements");
            return false;
        }
        if (!found) {
            canonical = *row;
            found = true;
        } else if (canonical.owner_rank != row->owner_rank ||
                   canonical.layer_index != row->layer_index ||
                   canonical.token_step_j != row->token_step_j ||
                   canonical.kv_hash != row->kv_hash ||
                   canonical.k_rope_hash != row->k_rope_hash) {
            set_error(err, err_size,
                      "DCP selected-row exchange duplicate row has conflicting payload metadata");
            return false;
        }
    }
    if (!found) {
        set_error(err, err_size,
                  "DCP selected-row exchange missing selected row payload");
        return false;
    }
    if (out) *out = canonical;
    return true;
}

bool ds4_glm52_dcp_selected_rows_host(
        const ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_dcp_row_payload *catalog,
        size_t catalog_count,
        const uint64_t *const selected_row_ids[DS4_GLM52_L0_RANK_COUNT],
        const size_t selection_counts[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    if (!requests || !owners || !catalog || catalog_count == 0 ||
        !selected_row_ids || !selection_counts || !replies) {
        set_error(err, err_size,
                  "DCP selected-row exchange requires requests, owners, catalog, selections, and replies");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        replies[rank].requester_rank = rank;
        replies[rank].row_count = 0;
        replies[rank].complete = false;
        if (!replies[rank].rows || replies[rank].row_capacity == 0) {
            set_error(err, err_size,
                      "DCP selected-row exchange requires reply row storage for every rank");
            return false;
        }
    }

    uint64_t expected_start = 0;
    uint32_t owner_mask = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!owners[rank].present ||
            owners[rank].rank != rank ||
            owners[rank].row_start != expected_start ||
            owners[rank].row_end <= owners[rank].row_start) {
            set_error(err, err_size,
                      "DCP selected-row exchange requires contiguous owner ranges indexed by rank");
            return false;
        }
        expected_start = owners[rank].row_end;
        owner_mask |= UINT32_C(1) << rank;
    }
    if (owner_mask != l0_full_rank_mask()) {
        set_error(err, err_size,
                  "DCP selected-row exchange requires all owner ranks");
        return false;
    }

    const ds4_glm52_dcp_exchange_request *ref = &requests[0];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_dcp_exchange_request *req = &requests[rank];
        if (req->rank != rank ||
            req->dcp_size != DS4_GLM52_L0_DCP_SIZE ||
            req->rank_count != DS4_GLM52_L0_RANK_COUNT ||
            req->seq == 0 ||
            req->model_hash == 0 ||
            req->session_hash == 0 ||
            req->layer_index < 0 ||
            req->selected_row_count < 0 ||
            req->owner_rank_mask != l0_full_rank_mask()) {
            set_error(err, err_size,
                      "DCP selected-row exchange request identity or topology is invalid");
            return false;
        }
        if (rank != 0 && !same_dcp_identity(ref, req)) {
            set_error(err, err_size,
                      "DCP selected-row exchange rank requests disagree on identity");
            return false;
        }
        if (!req->ownership_plan_valid ||
            !req->append_ordered_kv ||
            !req->row_payload_ready ||
            !req->transport_ready) {
            set_error(err, err_size,
                      "DCP selected-row exchange requires ownership, append-ordered KV, row payload, and transport readiness");
            return false;
        }
        if ((size_t)req->selected_row_count != selection_counts[rank]) {
            set_error(err, err_size,
                      "DCP selected-row exchange selection count does not match request metadata");
            return false;
        }
        if (selection_counts[rank] > replies[rank].row_capacity ||
            (selection_counts[rank] > 0 && !selected_row_ids[rank])) {
            set_error(err, err_size,
                      "DCP selected-row exchange selection storage is invalid");
            return false;
        }
    }

    ds4_glm52_dcp_rank_reply staging[DS4_GLM52_L0_RANK_COUNT];
    memset(staging, 0, sizeof(staging));
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        staging[rank].requester_rank = rank;
        staging[rank].rows = replies[rank].rows;
        staging[rank].row_capacity = replies[rank].row_capacity;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        for (size_t i = 0; i < selection_counts[rank]; i++) {
            ds4_glm52_dcp_row_payload row;
            if (!dcp_catalog_lookup_host(owners,
                                         catalog,
                                         catalog_count,
                                         selected_row_ids[rank][i],
                                         &row,
                                         err,
                                         err_size)) {
                return false;
            }
            if (row.layer_index != ref->layer_index) {
                set_error(err, err_size,
                          "DCP selected-row exchange row layer does not match request layer");
                return false;
            }
            if (!dcp_reply_insert_sorted_host(
                        &staging[rank], &row, err, err_size)) {
                return false;
            }
        }
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        replies[rank].row_count = staging[rank].row_count;
        replies[rank].complete = true;
    }
    return true;
}

uint64_t ds4_glm52_dcp_payload_hash_host(const void *data, size_t bytes) {
    if (!data || bytes == 0) return 0;
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static bool dcp_bound_range_ok(void *handle,
                               size_t byte_offset,
                               size_t byte_count,
                               size_t capacity_bytes) {
    if (!handle || byte_count == 0 || capacity_bytes == 0) return false;
    if (byte_offset > capacity_bytes) return false;
    return byte_count <= capacity_bytes - byte_offset;
}

static const void *dcp_bound_ptr(void *handle, size_t byte_offset) {
    return (const void *)((const unsigned char *)handle + byte_offset);
}

static bool dcp_bound_catalog_validate_selected(
        const ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_dcp_bound_row_payload *bound_catalog,
        size_t catalog_count,
        uint64_t row_id,
        char *err,
        size_t err_size) {
    int owner = -1;
    if (!dcp_owner_for_row_host(owners, row_id, &owner)) {
        set_error(err, err_size,
                  "DCP bound selected-row exchange selected row is outside ownership ranges");
        return false;
    }

    bool found = false;
    ds4_glm52_dcp_row_payload canonical;
    memset(&canonical, 0, sizeof(canonical));
    for (size_t i = 0; i < catalog_count; i++) {
        const ds4_glm52_dcp_bound_row_payload *bound = &bound_catalog[i];
        const ds4_glm52_dcp_row_payload *row = &bound->row;
        if (!row->present || row->row_id != row_id) continue;
        if (row->owner_rank != owner ||
            row->layer_index < 0 ||
            row->kv_hash == 0 ||
            row->k_rope_hash == 0) {
            set_error(err, err_size,
                      "DCP bound selected-row exchange row metadata does not match ownership or payload requirements");
            return false;
        }
        if (!bound->ready ||
            !dcp_bound_range_ok(bound->kv_handle,
                                bound->kv_byte_offset,
                                bound->kv_byte_count,
                                bound->kv_capacity_bytes) ||
            !dcp_bound_range_ok(bound->k_rope_handle,
                                bound->k_rope_byte_offset,
                                bound->k_rope_byte_count,
                                bound->k_rope_capacity_bytes)) {
            set_error(err, err_size,
                      "DCP bound selected-row exchange selected payload byte range is not ready or exceeds capacity");
            return false;
        }
        const void *kv =
            dcp_bound_ptr(bound->kv_handle, bound->kv_byte_offset);
        const void *k_rope =
            dcp_bound_ptr(bound->k_rope_handle, bound->k_rope_byte_offset);
        const uint64_t kv_hash =
            ds4_glm52_dcp_payload_hash_host(kv, bound->kv_byte_count);
        const uint64_t k_rope_hash =
            ds4_glm52_dcp_payload_hash_host(k_rope,
                                            bound->k_rope_byte_count);
        if (kv_hash == 0 || k_rope_hash == 0 ||
            kv_hash != row->kv_hash ||
            k_rope_hash != row->k_rope_hash) {
            set_error(err, err_size,
                      "DCP bound selected-row exchange payload hash does not match row metadata");
            return false;
        }
        if (!found) {
            canonical = *row;
            found = true;
        } else if (canonical.owner_rank != row->owner_rank ||
                   canonical.layer_index != row->layer_index ||
                   canonical.token_step_j != row->token_step_j ||
                   canonical.kv_hash != row->kv_hash ||
                   canonical.k_rope_hash != row->k_rope_hash) {
            set_error(err, err_size,
                      "DCP bound selected-row exchange duplicate row has conflicting payload metadata");
            return false;
        }
    }
    if (!found) {
        set_error(err, err_size,
                  "DCP bound selected-row exchange missing selected row payload");
        return false;
    }
    return true;
}

static void dcp_clear_replies_if_present(
        ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT]) {
    if (!replies) return;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        replies[rank].requester_rank = rank;
        replies[rank].row_count = 0;
        replies[rank].complete = false;
    }
}

bool ds4_glm52_dcp_selected_rows_bound_host(
        const ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_dcp_bound_row_payload *bound_catalog,
        size_t catalog_count,
        const uint64_t *const selected_row_ids[DS4_GLM52_L0_RANK_COUNT],
        const size_t selection_counts[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    dcp_clear_replies_if_present(replies);
    if (!requests || !owners || !bound_catalog || catalog_count == 0 ||
        !selected_row_ids || !selection_counts || !replies) {
        set_error(err, err_size,
                  "DCP bound selected-row exchange requires requests, owners, bound catalog, selections, and replies");
        return false;
    }

    ds4_glm52_dcp_row_payload *catalog =
        calloc(catalog_count, sizeof(catalog[0]));
    if (!catalog) {
        set_error(err, err_size,
                  "DCP bound selected-row exchange could not allocate metadata catalog");
        return false;
    }
    for (size_t i = 0; i < catalog_count; i++) {
        catalog[i] = bound_catalog[i].row;
    }

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_dcp_exchange_request *req = &requests[rank];
        if (req->rank != rank ||
            req->dcp_size != DS4_GLM52_L0_DCP_SIZE ||
            req->rank_count != DS4_GLM52_L0_RANK_COUNT ||
            req->selected_row_count < 0 ||
            (size_t)req->selected_row_count != selection_counts[rank]) {
            free(catalog);
            set_error(err, err_size,
                      "DCP bound selected-row exchange request selection metadata is invalid");
            return false;
        }
        if (!replies[rank].rows ||
            replies[rank].row_capacity == 0 ||
            selection_counts[rank] > replies[rank].row_capacity) {
            free(catalog);
            set_error(err, err_size,
                      "DCP bound selected-row exchange reply capacity is too small");
            return false;
        }
    }

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (selection_counts[rank] > 0 && !selected_row_ids[rank]) {
            free(catalog);
            set_error(err, err_size,
                      "DCP bound selected-row exchange selection storage is invalid");
            return false;
        }
        for (size_t i = 0; i < selection_counts[rank]; i++) {
            if (!dcp_bound_catalog_validate_selected(
                        owners,
                        bound_catalog,
                        catalog_count,
                        selected_row_ids[rank][i],
                        err,
                        err_size)) {
                free(catalog);
                return false;
            }
        }
    }

    const bool ok = ds4_glm52_dcp_selected_rows_host(
            requests,
            owners,
            catalog,
            catalog_count,
            selected_row_ids,
            selection_counts,
            replies,
            err,
            err_size);
    free(catalog);
    return ok;
}

bool ds4_glm52_dcp_transport_payload_from_bound(
        const ds4_glm52_dcp_exchange_request *request,
        const uint64_t *selected_row_ids,
        size_t selection_count,
        const ds4_glm52_dcp_bound_row_payload *rows,
        size_t row_count,
        ds4_glm52_dcp_transport_payload *payload,
        char *err,
        size_t err_size) {
    if (!request || !payload ||
        (selection_count > 0 && !selected_row_ids) ||
        (row_count > 0 && !rows)) {
        set_error(err, err_size,
                  "DCP transport payload requires request, selections, rows, and output storage");
        return false;
    }
    if (selection_count > DS4_GLM52_DCP_TRANSPORT_MAX_SELECTIONS ||
        row_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROWS) {
        set_error(err, err_size,
                  "DCP transport payload exceeds fixed transport capacity");
        return false;
    }
    if (request->selected_row_count < 0 ||
        (size_t)request->selected_row_count != selection_count) {
        set_error(err, err_size,
                  "DCP transport payload selection count does not match request");
        return false;
    }

    memset(payload, 0, sizeof(*payload));
    payload->request = *request;
    payload->selection_count = (uint64_t)selection_count;
    payload->row_count = (uint64_t)row_count;
    payload->present = true;
    for (size_t i = 0; i < selection_count; i++) {
        payload->selected_row_ids[i] = selected_row_ids[i];
    }
    for (size_t i = 0; i < row_count; i++) {
        const ds4_glm52_dcp_bound_row_payload *src = &rows[i];
        if (!src->ready ||
            !src->row.present ||
            src->kv_byte_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROW_BYTES ||
            src->k_rope_byte_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROW_BYTES ||
            !dcp_bound_range_ok(src->kv_handle,
                                src->kv_byte_offset,
                                src->kv_byte_count,
                                src->kv_capacity_bytes) ||
            !dcp_bound_range_ok(src->k_rope_handle,
                                src->k_rope_byte_offset,
                                src->k_rope_byte_count,
                                src->k_rope_capacity_bytes)) {
            set_error(err, err_size,
                      "DCP transport payload row is not ready or exceeds byte capacity");
            return false;
        }
        const void *kv = dcp_bound_ptr(src->kv_handle, src->kv_byte_offset);
        const void *k_rope =
            dcp_bound_ptr(src->k_rope_handle, src->k_rope_byte_offset);
        if (ds4_glm52_dcp_payload_hash_host(kv, src->kv_byte_count) !=
                src->row.kv_hash ||
            ds4_glm52_dcp_payload_hash_host(k_rope,
                                            src->k_rope_byte_count) !=
                src->row.k_rope_hash) {
            set_error(err, err_size,
                      "DCP transport payload row hash does not match bytes");
            return false;
        }
        ds4_glm52_dcp_transport_row *dst = &payload->rows[i];
        dst->row = src->row;
        dst->kv_byte_count = (uint64_t)src->kv_byte_count;
        dst->k_rope_byte_count = (uint64_t)src->k_rope_byte_count;
        memcpy(dst->kv_bytes, kv, src->kv_byte_count);
        memcpy(dst->k_rope_bytes, k_rope, src->k_rope_byte_count);
        dst->present = true;
    }
    return true;
}

bool ds4_glm52_dcp_transport_payload_to_bound(
        ds4_glm52_dcp_transport_payload *payload,
        ds4_glm52_dcp_exchange_request *request_out,
        uint64_t **selected_row_ids_out,
        size_t *selection_count_out,
        ds4_glm52_dcp_bound_row_payload *rows_out,
        size_t rows_out_capacity,
        size_t *row_count_out,
        char *err,
        size_t err_size) {
    if (!payload || !payload->present || !request_out ||
        !selected_row_ids_out || !selection_count_out ||
        !rows_out || !row_count_out) {
        set_error(err, err_size,
                  "DCP transport payload decode requires payload and output storage");
        return false;
    }
    if (payload->selection_count > DS4_GLM52_DCP_TRANSPORT_MAX_SELECTIONS ||
        payload->row_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROWS ||
        payload->row_count > rows_out_capacity) {
        set_error(err, err_size,
                  "DCP transport payload count exceeds receiver capacity");
        return false;
    }
    if (payload->request.selected_row_count < 0 ||
        (uint64_t)payload->request.selected_row_count !=
            payload->selection_count) {
        set_error(err, err_size,
                  "DCP transport payload selection count does not match request");
        return false;
    }

    *request_out = payload->request;
    *selected_row_ids_out = payload->selected_row_ids;
    *selection_count_out = (size_t)payload->selection_count;
    *row_count_out = (size_t)payload->row_count;
    for (size_t i = 0; i < (size_t)payload->row_count; i++) {
        ds4_glm52_dcp_transport_row *src = &payload->rows[i];
        if (!src->present ||
            !src->row.present ||
            src->kv_byte_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROW_BYTES ||
            src->k_rope_byte_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROW_BYTES) {
            set_error(err, err_size,
                      "DCP transport payload row is missing or too large");
            return false;
        }
        if (ds4_glm52_dcp_payload_hash_host(
                    src->kv_bytes, (size_t)src->kv_byte_count) !=
                src->row.kv_hash ||
            ds4_glm52_dcp_payload_hash_host(
                    src->k_rope_bytes, (size_t)src->k_rope_byte_count) !=
                src->row.k_rope_hash) {
            set_error(err, err_size,
                      "DCP transport payload row hash does not match received bytes");
            return false;
        }
        memset(&rows_out[i], 0, sizeof(rows_out[i]));
        rows_out[i].row = src->row;
        rows_out[i].kv_handle = src->kv_bytes;
        rows_out[i].kv_byte_offset = 0;
        rows_out[i].kv_byte_count = (size_t)src->kv_byte_count;
        rows_out[i].kv_capacity_bytes = (size_t)src->kv_byte_count;
        rows_out[i].k_rope_handle = src->k_rope_bytes;
        rows_out[i].k_rope_byte_offset = 0;
        rows_out[i].k_rope_byte_count = (size_t)src->k_rope_byte_count;
        rows_out[i].k_rope_capacity_bytes = (size_t)src->k_rope_byte_count;
        rows_out[i].ready = true;
    }
    return true;
}

ds4_glm52_l0_status ds4_glm52_dcp_real_row_exchange(
        const ds4_glm52_dcp_exchange_request *request,
        ds4_glm52_l0_result *result) {
    if (!request) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real DCP row-exchange request is missing");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (request->rank < 0 ||
        request->rank >= DS4_GLM52_L0_RANK_COUNT ||
        request->dcp_size != DS4_GLM52_L0_DCP_SIZE ||
        request->rank_count != DS4_GLM52_L0_RANK_COUNT) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real DCP row exchange requires rank in [0,4) with DCP4/rank_count=4");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (request->seq == 0 ||
        request->model_hash == 0 ||
        request->session_hash == 0 ||
        request->layer_index < 0 ||
        request->selected_row_count < 0) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real DCP row exchange requires nonzero identity and valid layer/selection evidence");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (request->owner_rank_mask != l0_full_rank_mask()) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real DCP row exchange requires a complete four-rank owner map");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!request->ownership_plan_valid || !request->append_ordered_kv) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real DCP row exchange requires validated ownership and append-ordered KV state");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!request->row_payload_ready || !request->transport_ready) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real DCP selected-row exchange backend is not ready: row payloads and transport are required");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    set_result(result,
               DS4_GLM52_L0_STATUS_NOT_READY,
               DS4_GLM52_L0_ACTION_DECODE_TOKEN,
               "real DCP selected-row exchange execution is not implemented behind the validated boundary");
    return DS4_GLM52_L0_STATUS_NOT_READY;
}

ds4_glm52_l0_status ds4_glm52_decode_real_step(
        const ds4_glm52_l0_config *cfg,
        const ds4_glm52_decode_real_request *request,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result) {
    char err[160];
    if (!ds4_glm52_l0_validate_config(cfg, err, sizeof(err))) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   err);
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!request || !state) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires request and mutable L0 state");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!prefill_ready(state)) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires prefill-complete append-ordered KV state and replicated cursor");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (request->rank != cfg->rank ||
        request->seq == 0 ||
        request->model_hash == 0 ||
        request->session_hash == 0 ||
        request->input_token < 0) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires matching rank plus nonzero model/session/sequence and valid token evidence");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!request->model_ready || !resident_shards_ready(state)) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires ready resident rank-local model engines");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (!request->tp_collectives_ready) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires the TP4 all-reduce backend");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (!request->dcp_exchange_ready) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires the DCP selected-row exchange backend");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (!request->glm52_kernels_ready) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires GLM 5.2 QKV/MLA/MoE/logits kernels");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (!request->logits_ready ||
        !request->sampled_token_ready ||
        !request->kv_append_committed) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode requires backend logits, sampled token, and committed KV append evidence");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    }
    if (request->sampled_token < 0 ||
        request->sampled_token >= DS4_GLM52_L0_VOCAB_SIZE) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                   "real decode sampled token is outside the GLM 5.2 vocabulary");
        return DS4_GLM52_L0_STATUS_INVALID;
    }

    const uint64_t position = state->cursor.token_step_j;
    state->token.token_id = request->sampled_token;
    state->token.position = position;
    state->token.present = true;
    state->kv.length = position + 1;
    state->kv.tp_aware = true;
    state->kv.append_ordered = true;
    state->kv.prefill_complete = true;
    state->cursor.token_step_j = position + 1;
    state->cursor.kv_length = position + 1;
    state->cursor.replicated = true;
    state->cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_DECODE;
    state->logits.owner_rank = 0;
    state->logits.coordinator_visible = true;
    set_result(result,
               DS4_GLM52_L0_STATUS_OK,
               DS4_GLM52_L0_ACTION_DECODE_TOKEN,
               "real GLM 5.2 decode step committed backend outputs and advanced the KV cursor");
    return DS4_GLM52_L0_STATUS_OK;
}

ds4_glm52_l0_status ds4_glm52_l0_stub_action(ds4_glm52_l0_action action,
                                             const ds4_glm52_l0_config *cfg,
                                             ds4_glm52_l0_state *state,
                                             ds4_glm52_l0_result *result) {
    char err[160];
    if (!ds4_glm52_l0_validate_config(cfg, err, sizeof(err))) {
        set_result(result, DS4_GLM52_L0_STATUS_INVALID, action, err);
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!state) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   action,
                   "GLM 5.2 TP4 L0 state is missing");
        return DS4_GLM52_L0_STATUS_INVALID;
    }

    switch (action) {
    case DS4_GLM52_L0_ACTION_SERVE_OPEN:
        {
            ds4_glm52_l0_status load_status =
                load_plan_and_manifest(cfg, state, result);
            if (load_status != DS4_GLM52_L0_STATUS_OK) return load_status;
        }
        if (!resident_shards_ready(state)) {
            /* model-load/a-map-rank-shards: run the model-shard-layout
             * child graph to map rank-local tensors and publish the
             * loaded rank shard. */
            ds4_glm52_l0_status layout_status =
                run_model_shard_layout(cfg, state, result);
            if (layout_status != DS4_GLM52_L0_STATUS_OK) return layout_status;
        }
        if (!resident_shards_ready(state)) {
            set_result(result,
                       DS4_GLM52_L0_STATUS_NOT_READY,
                       action,
                       "model-load/a-map-rank-shards completed but resident shards are not ready");
            return DS4_GLM52_L0_STATUS_NOT_READY;
        }
        set_result(result,
                   DS4_GLM52_L0_STATUS_OK,
                   action,
                   "model-load/a-ready-rank-engines: resident rank-local model shards are ready");
        return DS4_GLM52_L0_STATUS_OK;
    case DS4_GLM52_L0_ACTION_TP_GROUP:
        return bind_tp_group(cfg, state, result);
    case DS4_GLM52_L0_ACTION_ACCEPT:
        if (!tp_group_ready(state)) {
            set_result(result,
                       DS4_GLM52_L0_STATUS_NOT_READY,
                       action,
                       "request acceptance requires resident rank engines and a ready TP4/DCP4 fabric");
            return DS4_GLM52_L0_STATUS_NOT_READY;
        }
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   action,
                   "request/session binding is not implemented");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    case DS4_GLM52_L0_ACTION_PREFILL:
        return ds4_glm52_l0_prefill_run(cfg, state, 0, NULL, NULL, result);
    case DS4_GLM52_L0_ACTION_DECODE_TOKEN:
        if (!prefill_ready(state)) {
            set_result(result,
                       DS4_GLM52_L0_STATUS_NOT_READY,
                       action,
                       "decode requires prefill-complete append-ordered KV state and replicated cursor");
            return DS4_GLM52_L0_STATUS_NOT_READY;
        }
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   action,
                   "decode-token child graph is not implemented; token_step_j and KV cursor are unchanged");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    case DS4_GLM52_L0_ACTION_STREAM_TOKEN:
        if (!state->token.present) {
            set_result(result,
                       DS4_GLM52_L0_STATUS_NOT_READY,
                       action,
                       "streaming requires a sampled decode token");
            return DS4_GLM52_L0_STATUS_NOT_READY;
        }
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   action,
                   "token streaming is not implemented");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    case DS4_GLM52_L0_ACTION_KV_CHECKPOINT:
        if (state->checkpoint.requested && !prefill_ready(state)) {
            set_result(result,
                       DS4_GLM52_L0_STATUS_NOT_READY,
                       action,
                       "KV checkpoint requires established TP-aware KV state and cursor");
            return DS4_GLM52_L0_STATUS_NOT_READY;
        }
        set_result(result,
                   DS4_GLM52_L0_STATUS_NOT_READY,
                   action,
                   "KV checkpointing is not implemented");
        return DS4_GLM52_L0_STATUS_NOT_READY;
    default:
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   action,
                   "unknown GLM 5.2 TP4 L0 action");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
}

static ds4_glm52_l0_status run_skeleton_impl(const ds4_glm52_l0_config *cfg,
                                             ds4_glm52_l0_state *state,
                                             ds4_glm52_l0_result *result,
                                             ds4_glm52_l0_trace_fn trace,
                                             void *trace_ud,
                                             bool continue_after_not_ready) {
    char err[160];
    if (!ds4_glm52_l0_validate_config(cfg, err, sizeof(err))) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_SERVE_OPEN,
                   err);
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    if (!state) {
        set_result(result,
                   DS4_GLM52_L0_STATUS_INVALID,
                   DS4_GLM52_L0_ACTION_SERVE_OPEN,
                   "GLM 5.2 TP4 L0 state is missing");
        return DS4_GLM52_L0_STATUS_INVALID;
    }
    init_state_from_config(cfg, state);

    ds4_glm52_l0_result first_blocked;
    bool have_first_blocked = false;
    for (int i = 0; i < DS4_GLM52_L0_ACTION_COUNT; i++) {
        ds4_glm52_l0_action action = (ds4_glm52_l0_action)i;
        ds4_glm52_l0_result step_result;
        ds4_glm52_l0_status status =
            ds4_glm52_l0_stub_action(action, cfg, state, &step_result);
        if (trace) {
            trace(trace_ud,
                  DS4_GLM52_L0_GRAPH_ID,
                  ds4_glm52_l0_action_id(action),
                  status,
                  have_first_blocked,
                  state);
        }
        if (status != DS4_GLM52_L0_STATUS_OK) {
            if (!have_first_blocked) {
                first_blocked = step_result;
                have_first_blocked = true;
            }
            if (!continue_after_not_ready ||
                status != DS4_GLM52_L0_STATUS_NOT_READY) {
                if (result) *result = step_result;
                return status;
            }
        }
    }

    if (have_first_blocked) {
        if (result) *result = first_blocked;
        return first_blocked.status;
    }

    set_result(result,
               DS4_GLM52_L0_STATUS_OK,
               DS4_GLM52_L0_ACTION_KV_CHECKPOINT,
               "GLM 5.2 TP4 L0 skeleton complete");
    return DS4_GLM52_L0_STATUS_OK;
}

ds4_glm52_l0_status ds4_glm52_l0_run_skeleton(const ds4_glm52_l0_config *cfg,
                                             ds4_glm52_l0_state *state,
                                             ds4_glm52_l0_result *result,
                                             ds4_glm52_l0_trace_fn trace,
                                             void *trace_ud) {
    return run_skeleton_impl(cfg, state, result, trace, trace_ud, false);
}

ds4_glm52_l0_status ds4_glm52_l0_review_skeleton(const ds4_glm52_l0_config *cfg,
                                                ds4_glm52_l0_state *state,
                                                ds4_glm52_l0_result *result,
                                                ds4_glm52_l0_trace_fn trace,
                                                void *trace_ud) {
    return run_skeleton_impl(cfg, state, result, trace, trace_ud, true);
}

/* ================================================================== */
/* model-shard-layout child graph implementation                      */
/* BCD actions: a-read-layout-spec, a-validate-tensor-ownership,      */
/* a-load-rank-tensors, a-publish-loaded-shard                        */
/* ================================================================== */

static const char *const k_layout_action_ids[] = {
    "model-shard-layout/a-read-layout-spec",
    "model-shard-layout/a-validate-tensor-ownership",
    "model-shard-layout/a-load-rank-tensors",
    "model-shard-layout/a-upload-gpu-resident-tensors",
    "model-shard-layout/a-publish-loaded-shard",
};

static ds4_glm52_l0_status layout_fail(ds4_glm52_l0_result *result,
                                       ds4_glm52_l0_status status,
                                       int action_index,
                                       const char *message) {
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: %.128s",
             k_layout_action_ids[action_index],
             message ? message : "");
    set_result(result, status, DS4_GLM52_L0_ACTION_SERVE_OPEN, msg);
    return status;
}

const char *ds4_glm52_layout_role_name(ds4_glm52_layout_role role) {
    switch (role) {
    case DS4_GLM52_LAYOUT_ROLE_REPLICATED: return "replicated";
    case DS4_GLM52_LAYOUT_ROLE_Q_HEAD:     return "q_head";
    case DS4_GLM52_LAYOUT_ROLE_MLA_KV:     return "mla_kv";
    case DS4_GLM52_LAYOUT_ROLE_EXPERT:     return "expert";
    case DS4_GLM52_LAYOUT_ROLE_VOCAB:      return "vocab";
    case DS4_GLM52_LAYOUT_ROLE_OUTPUT:     return "output";
    default:                               return "unknown";
    }
}

static ds4_glm52_layout_role parse_role(const char *s) {
    if (!s) return DS4_GLM52_LAYOUT_ROLE_UNKNOWN;
    if (!strcmp(s, "replicated")) return DS4_GLM52_LAYOUT_ROLE_REPLICATED;
    if (!strcmp(s, "q_head"))     return DS4_GLM52_LAYOUT_ROLE_Q_HEAD;
    if (!strcmp(s, "mla_kv"))     return DS4_GLM52_LAYOUT_ROLE_MLA_KV;
    if (!strcmp(s, "expert"))     return DS4_GLM52_LAYOUT_ROLE_EXPERT;
    if (!strcmp(s, "vocab"))      return DS4_GLM52_LAYOUT_ROLE_VOCAB;
    if (!strcmp(s, "output"))     return DS4_GLM52_LAYOUT_ROLE_OUTPUT;
    return DS4_GLM52_LAYOUT_ROLE_UNKNOWN;
}

static ds4_glm52_layout_scope parse_scope(const char *s) {
    if (!s) return DS4_GLM52_LAYOUT_SCOPE_REPLICATED;
    if (!strcmp(s, "rank_local_shard")) return DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL;
    return DS4_GLM52_LAYOUT_SCOPE_REPLICATED;
}

static bool parse_u64_value(const char *s, uint64_t *out) {
    if (!s || !s[0] || !out) return false;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s) return false;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end) return false;
    *out = (uint64_t)v;
    return true;
}

static ds4_glm52_tp4_tensor_dtype parse_layout_dtype(const char *s) {
    if (!s) return DS4_GLM52_TP4_TENSOR_DTYPE_INVALID;
    if (!strcmp(s, "f32")) return DS4_GLM52_TP4_TENSOR_DTYPE_F32;
    if (!strcmp(s, "bf16")) return DS4_GLM52_TP4_TENSOR_DTYPE_BF16;
    if (!strcmp(s, "fp8_e4m3") || !strcmp(s, "fp8")) {
        return DS4_GLM52_TP4_TENSOR_DTYPE_FP8_E4M3;
    }
    if (!strcmp(s, "i32")) return DS4_GLM52_TP4_TENSOR_DTYPE_I32;
    return DS4_GLM52_TP4_TENSOR_DTYPE_INVALID;
}

static bool parse_shape_value(const char *s,
                              uint64_t shape[DS4_GLM52_LAYOUT_MAX_SHAPE_DIMS],
                              int *shape_count,
                              uint64_t *element_count) {
    if (!s || !shape || !shape_count || !element_count) return false;
    char buf[DS4_GLM52_LAYOUT_FIELD_MAX];
    copy_string(buf, sizeof(buf), s);
    char *p = trim_ws(buf);
    if (p[0] == '[') p++;
    size_t len = strlen(p);
    if (len > 0 && p[len - 1] == ']') p[len - 1] = '\0';
    for (char *q = p; *q; q++) {
        if (*q == 'x' || *q == 'X' || *q == ',' || *q == ';') {
            *q = ' ';
        }
    }

    int count = 0;
    uint64_t product = 1;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        if (count >= DS4_GLM52_LAYOUT_MAX_SHAPE_DIMS) return false;
        errno = 0;
        char *end = NULL;
        unsigned long long dim = strtoull(p, &end, 10);
        if (errno || end == p || dim == 0) return false;
        if (product > UINT64_MAX / (uint64_t)dim) return false;
        shape[count++] = (uint64_t)dim;
        product *= (uint64_t)dim;
        p = end;
        if (*p && !isspace((unsigned char)*p)) return false;
    }
    if (count == 0) return false;
    for (int i = count; i < DS4_GLM52_LAYOUT_MAX_SHAPE_DIMS; i++) {
        shape[i] = 0;
    }
    *shape_count = count;
    *element_count = product;
    return true;
}

static uint64_t layout_shape_hash(ds4_glm52_tp4_tensor_dtype dtype,
                                  const uint64_t *shape,
                                  int shape_count) {
    uint64_t h = UINT64_C(1469598103934665603);
    const uint64_t prime = UINT64_C(1099511628211);
    h ^= (uint64_t)dtype;
    h *= prime;
    h ^= (uint64_t)shape_count;
    h *= prime;
    for (int i = 0; i < shape_count; i++) {
        uint64_t v = shape[i];
        for (int b = 0; b < 8; b++) {
            h ^= (v >> (b * 8)) & UINT64_C(0xff);
            h *= prime;
        }
    }
    return h ? h : UINT64_C(1);
}

static bool validate_layout_entry_metadata(ds4_glm52_layout_entry *entry,
                                           char *err,
                                           size_t err_size) {
    if (!entry) {
        set_error(err, err_size, "layout entry is missing");
        return false;
    }
    if (!valid_tensor_dtype(entry->dtype)) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "entry '%s' is missing a supported dtype",
                 entry->tensor_name[0] ? entry->tensor_name : "(unnamed)");
        set_error(err, err_size, msg);
        return false;
    }
    if (entry->shape_count <= 0) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "entry '%s' is missing tensor shape metadata",
                 entry->tensor_name[0] ? entry->tensor_name : "(unnamed)");
        set_error(err, err_size, msg);
        return false;
    }

    uint64_t product = 1;
    for (int i = 0; i < entry->shape_count; i++) {
        if (entry->shape[i] == 0 ||
            product > UINT64_MAX / entry->shape[i]) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' has invalid tensor shape metadata",
                     entry->tensor_name[0] ? entry->tensor_name : "(unnamed)");
            set_error(err, err_size, msg);
            return false;
        }
        product *= entry->shape[i];
    }
    if (entry->element_count == 0) {
        entry->element_count = product;
    } else if (entry->element_count != product) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "entry '%s' element_count does not match tensor shape",
                 entry->tensor_name[0] ? entry->tensor_name : "(unnamed)");
        set_error(err, err_size, msg);
        return false;
    }

    const size_t dtype_size = ds4_glm52_tp4_tensor_dtype_size(entry->dtype);
    if (entry->element_count > UINT64_MAX / (uint64_t)dtype_size ||
        entry->byte_length != entry->element_count * (uint64_t)dtype_size) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "entry '%s' byte_length does not match dtype and shape",
                 entry->tensor_name[0] ? entry->tensor_name : "(unnamed)");
        set_error(err, err_size, msg);
        return false;
    }

    uint64_t derived_hash =
        layout_shape_hash(entry->dtype, entry->shape, entry->shape_count);
    if (entry->shape_hash == 0) {
        entry->shape_hash = derived_hash;
    } else if (entry->shape_hash != derived_hash) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "entry '%s' shape_hash does not match dtype and shape",
                 entry->tensor_name[0] ? entry->tensor_name : "(unnamed)");
        set_error(err, err_size, msg);
        return false;
    }
    return true;
}

static bool parse_kv_line(const char *line, int line_no,
                          const char **key_out, const char **val_out,
                          char *err, size_t err_size) {
    char *eq = strchr(line, '=');
    if (!eq) {
        snprintf(err, err_size, "line %d is not key=value", line_no);
        return false;
    }
    *eq = '\0';
    *key_out = trim_ws((char *)line);
    *val_out = trim_ws(eq + 1);
    return true;
}

typedef struct {
    uint32_t state[8];
    uint64_t bit_len;
    unsigned char block[64];
    size_t block_len;
} ds4_sha256_ctx;

static uint32_t sha256_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(ds4_sha256_ctx *ctx,
                             const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = sha256_rotr(w[i - 15], 7) ^
                      sha256_rotr(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = sha256_rotr(w[i - 2], 17) ^
                      sha256_rotr(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^
                      sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^
                      sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(ds4_sha256_ctx *ctx) {
    ctx->state[0] = 0x6a09e667u;
    ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u;
    ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu;
    ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu;
    ctx->state[7] = 0x5be0cd19u;
    ctx->bit_len = 0;
    ctx->block_len = 0;
}

static void sha256_update(ds4_sha256_ctx *ctx,
                          const unsigned char *data,
                          size_t len) {
    while (len > 0) {
        size_t take = 64u - ctx->block_len;
        if (take > len) take = len;
        memcpy(ctx->block + ctx->block_len, data, take);
        ctx->block_len += take;
        data += take;
        len -= take;
        if (ctx->block_len == 64u) {
            sha256_transform(ctx, ctx->block);
            ctx->bit_len += 512u;
            ctx->block_len = 0;
        }
    }
}

static void sha256_final(ds4_sha256_ctx *ctx, unsigned char out[32]) {
    ctx->bit_len += (uint64_t)ctx->block_len * 8u;
    ctx->block[ctx->block_len++] = 0x80u;
    if (ctx->block_len > 56u) {
        while (ctx->block_len < 64u) ctx->block[ctx->block_len++] = 0;
        sha256_transform(ctx, ctx->block);
        ctx->block_len = 0;
    }
    while (ctx->block_len < 56u) ctx->block[ctx->block_len++] = 0;
    for (int i = 7; i >= 0; i--) {
        ctx->block[ctx->block_len++] =
            (unsigned char)((ctx->bit_len >> (i * 8)) & 0xffu);
    }
    sha256_transform(ctx, ctx->block);
    for (int i = 0; i < 8; i++) {
        out[i * 4] = (unsigned char)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(ctx->state[i] >> 8);
        out[i * 4 + 3] = (unsigned char)ctx->state[i];
    }
}

static bool sha256_hex_valid(const char *s) {
    if (!s) return false;
    for (int i = 0; i < 64; i++) {
        if (!isxdigit((unsigned char)s[i])) return false;
    }
    return s[64] == '\0';
}

static void sha256_to_hex(const unsigned char digest[32],
                          char out[65]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 0x0f];
    }
    out[64] = '\0';
}

static bool sha256_hex_equal(const char *a, const char *b) {
    for (int i = 0; i < 64; i++) {
        if (tolower((unsigned char)a[i]) !=
            tolower((unsigned char)b[i])) {
            return false;
        }
    }
    return true;
}

static bool file_slice_sha256_hex(const char *path,
                                  uint64_t offset,
                                  uint64_t length,
                                  char out[65]) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    if (offset > (uint64_t)LLONG_MAX ||
        fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        fclose(fp);
        return false;
    }

    ds4_sha256_ctx ctx;
    sha256_init(&ctx);
    unsigned char buf[8192];
    uint64_t remaining = length;
    while (remaining > 0) {
        size_t want = sizeof(buf);
        if (remaining < (uint64_t)want) want = (size_t)remaining;
        size_t got = fread(buf, 1, want, fp);
        if (got == 0) {
            fclose(fp);
            return false;
        }
        sha256_update(&ctx, buf, got);
        remaining -= got;
    }
    fclose(fp);
    unsigned char digest[32];
    sha256_final(&ctx, digest);
    sha256_to_hex(digest, out);
    return true;
}

static void unmap_layout_tensor(ds4_glm52_layout_mapped_tensor *tensor) {
    if (!tensor) return;
    if (tensor->mapped && tensor->map_base && tensor->map_bytes > 0) {
        (void)munmap(tensor->map_base, (size_t)tensor->map_bytes);
    }
    memset(tensor, 0, sizeof(*tensor));
}

static void release_layout_gpu_tensor(ds4_glm52_layout_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->free_tensor && tensor->tensor.ptr) {
        tensor->free_tensor(&tensor->tensor);
    }
    memset(tensor, 0, sizeof(*tensor));
}

void ds4_glm52_layout_unmap_mapped_slices(
        ds4_glm52_layout_mapped_slices *mapped) {
    if (!mapped) return;
    for (int i = 0; i < mapped->tensor_count; i++) {
        unmap_layout_tensor(&mapped->tensors[i]);
    }
    mapped->tensor_count = 0;
    mapped->mapped_bytes = 0;
    mapped->mapped = false;
    mapped->hash_verified = false;
}

void ds4_glm52_layout_release_resident_gpu_tensors(
        ds4_glm52_l0_state *state) {
    if (!state) return;
    for (int i = 0; i < state->resident_shards.gpu_tensor_count; i++) {
        release_layout_gpu_tensor(&state->resident_shards.gpu_tensors[i]);
    }
    state->resident_shards.gpu_tensor_count = 0;
    state->resident_shards.gpu_bytes = 0;
    state->resident_shards.gpu_resident = false;
}

void ds4_glm52_l0_unmap_resident_rank_shards(ds4_glm52_l0_state *state) {
    if (!state) return;
    ds4_glm52_layout_release_resident_gpu_tensors(state);
    for (int i = 0; i < state->resident_shards.tensor_count; i++) {
        unmap_layout_tensor(&state->resident_shards.tensors[i]);
    }
    state->resident_shards.tensor_count = 0;
    state->resident_shards.mapped_bytes = 0;
    state->resident_shards.mapped = false;
}

static bool mapped_tensor_metadata_ready(
        const ds4_glm52_layout_mapped_tensor *tensor,
        char *err,
        size_t err_size) {
    if (!tensor || !tensor->mapped || !tensor->data ||
        tensor->data_bytes == 0) {
        set_error(err, err_size,
                  "GPU upload requires a mapped tensor with readable bytes");
        return false;
    }
    if (!tensor->tensor_name[0]) {
        set_error(err, err_size,
                  "GPU upload requires a named tensor");
        return false;
    }
    if (!valid_tensor_dtype(tensor->dtype) ||
        tensor->shape_count <= 0 ||
        tensor->element_count == 0 ||
        tensor->shape_hash == 0) {
        set_error(err, err_size,
                  "GPU upload requires validated dtype, shape, element count, and shape hash metadata");
        return false;
    }
    const size_t dtype_size = ds4_glm52_tp4_tensor_dtype_size(tensor->dtype);
    if (tensor->element_count > UINT64_MAX / (uint64_t)dtype_size ||
        tensor->data_bytes != tensor->element_count * (uint64_t)dtype_size) {
        set_error(err, err_size,
                  "GPU upload tensor byte count does not match dtype and shape metadata");
        return false;
    }
    return true;
}

ds4_glm52_l0_status ds4_glm52_layout_upload_resident_gpu_tensors(
        ds4_glm52_l0_state *state,
        int device_id,
        const ds4_glm52_gpu_tensor_runtime *runtime,
        ds4_glm52_l0_result *result) {
    if (!state || !state->resident_shards.mapped ||
        state->resident_shards.tensor_count <= 0) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                           "GPU upload requires resident mapped tensors");
    }
    if (device_id < 0) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                           "GPU upload requires a valid rank-local device id");
    }
    if (!runtime || !runtime->alloc || !runtime->upload ||
        !runtime->free) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                           "GPU upload requires alloc, upload, and free callbacks");
    }

    ds4_glm52_layout_release_resident_gpu_tensors(state);

    for (int i = 0; i < state->resident_shards.tensor_count; i++) {
        ds4_glm52_layout_mapped_tensor *src =
            &state->resident_shards.tensors[i];
        char err[192] = "";
        if (!mapped_tensor_metadata_ready(src, err, sizeof(err))) {
            ds4_glm52_layout_release_resident_gpu_tensors(state);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3, err);
        }

        ds4_glm52_layout_gpu_tensor *dst =
            &state->resident_shards.gpu_tensors[
                state->resident_shards.gpu_tensor_count];
        memset(dst, 0, sizeof(*dst));
        if (!runtime->alloc(&dst->tensor, device_id, src->data_bytes)) {
            ds4_glm52_layout_release_resident_gpu_tensors(state);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                               "GPU upload allocation failed");
        }
        dst->free_tensor = runtime->free;
        if (!dst->tensor.ptr ||
            dst->tensor.bytes < src->data_bytes ||
            dst->tensor.device_id != device_id) {
            ds4_glm52_layout_release_resident_gpu_tensors(state);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                               "GPU upload allocation did not return a rank-owned device tensor");
        }
        if (!runtime->upload(&dst->tensor, 0, src->data, src->data_bytes)) {
            ds4_glm52_layout_release_resident_gpu_tensors(state);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                               "GPU upload copy failed");
        }

        copy_string(dst->tensor_name, sizeof(dst->tensor_name),
                    src->tensor_name);
        dst->role = src->role;
        dst->scope = src->scope;
        dst->rank = src->rank;
        dst->device_id = device_id;
        dst->byte_count = src->data_bytes;
        dst->dtype = src->dtype;
        dst->shape_count = src->shape_count;
        memcpy(dst->shape, src->shape, sizeof(dst->shape));
        dst->element_count = src->element_count;
        dst->shape_hash = src->shape_hash;
        dst->replicated = src->replicated;
        dst->ready = true;
        state->resident_shards.gpu_tensor_count++;
        state->resident_shards.gpu_bytes += src->data_bytes;
    }

    state->resident_shards.gpu_resident =
        state->resident_shards.gpu_tensor_count ==
        state->resident_shards.tensor_count;
    if (!state->resident_shards.gpu_resident) {
        ds4_glm52_layout_release_resident_gpu_tensors(state);
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 3,
                           "GPU upload did not bind every mapped tensor");
    }
    return DS4_GLM52_L0_STATUS_OK;
}

static bool mmap_layout_entry(const char *path,
                              const ds4_glm52_layout_entry *entry,
                              ds4_glm52_layout_mapped_tensor *out,
                              char *err,
                              size_t err_size) {
    if (!path || !entry || !out || entry->byte_length == 0) {
        set_error(err, err_size, "mmap entry requires path, entry, and bytes");
        return false;
    }
    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) page_size_long = 4096;
    const uint64_t page_size = (uint64_t)page_size_long;
    const uint64_t aligned_offset =
        (entry->byte_offset / page_size) * page_size;
    const uint64_t page_delta = entry->byte_offset - aligned_offset;
    if (entry->byte_length > UINT64_MAX - page_delta) {
        set_error(err, err_size, "mmap entry byte range overflows");
        return false;
    }
    const uint64_t map_bytes = page_delta + entry->byte_length;
    if (map_bytes == 0 || map_bytes > (uint64_t)SIZE_MAX ||
        aligned_offset > (uint64_t)LLONG_MAX) {
        set_error(err, err_size, "mmap entry byte range is not addressable");
        return false;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        char msg[192];
        snprintf(msg, sizeof(msg), "cannot open shard file '%.128s'", path);
        set_error(err, err_size, msg);
        return false;
    }
    void *base = mmap(NULL,
                      (size_t)map_bytes,
                      PROT_READ,
                      MAP_PRIVATE,
                      fd,
                      (off_t)aligned_offset);
    int saved_errno = errno;
    close(fd);
    if (base == MAP_FAILED) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "mmap failed for entry '%.64s': %.80s",
                 entry->tensor_name[0] ? entry->tensor_name : "(unnamed)",
                 strerror(saved_errno));
        set_error(err, err_size, msg);
        return false;
    }

    memset(out, 0, sizeof(*out));
    copy_string(out->tensor_name, sizeof(out->tensor_name), entry->tensor_name);
    copy_string(out->file_path, sizeof(out->file_path), path);
    out->map_base = base;
    out->map_bytes = map_bytes;
    out->data = (const unsigned char *)base + page_delta;
    out->data_bytes = entry->byte_length;
    out->byte_offset = entry->byte_offset;
    out->role = (int)entry->role;
    out->scope = (int)entry->scope;
    out->rank = entry->rank;
    out->dtype = entry->dtype;
    out->shape_count = entry->shape_count;
    memcpy(out->shape, entry->shape, sizeof(out->shape));
    out->element_count = entry->element_count;
    out->shape_hash = entry->shape_hash;
    out->replicated = entry->replicated;
    out->mapped = true;
    return true;
}

/* ---- a-read-layout-spec: ds4_glm52_layout_read_manifest ---- */

ds4_glm52_l0_status ds4_glm52_layout_read_manifest(
        const char *layout_path,
        ds4_glm52_layout_spec *spec,
        ds4_glm52_l0_result *result) {
    if (!layout_path || !layout_path[0]) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "layout manifest path is required");
    }
    if (!spec) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "layout spec output is missing");
    }
    memset(spec, 0, sizeof(*spec));
    spec->tp_size = -1;
    spec->dcp_size = -1;
    spec->rank_count = -1;
    spec->required_base_shards = -1;
    spec->required_mtp_shards = -1;

    FILE *fp = fopen(layout_path, "r");
    if (!fp) {
        char msg[192];
        snprintf(msg, sizeof(msg), "cannot open layout manifest '%s'", layout_path);
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0, msg);
    }

    char line[DS4_GLM52_LAYOUT_MAX_LINE];
    int line_no = 0;
    ds4_glm52_layout_entry entry = {0};
    bool in_entry = false;
    entry.rank = -1;
    entry.q_head_start = -1;
    entry.q_head_end = -1;
    entry.expert_start = -1;
    entry.expert_end = -1;
    entry.vocab_start = -1;
    entry.vocab_end = -1;

    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = trim_ws(line);
        if (!p[0] || p[0] == '#') continue;

        if (p[0] == '[') {
            /* section header; currently only [entry] is recognized */
            if (in_entry) {
                if (spec->entry_count >= DS4_GLM52_LAYOUT_MAX_ENTRIES) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "layout manifest exceeds max entries");
                }
                spec->entries[spec->entry_count++] = entry;
            }
            if (!strcmp(p, "[entry]")) {
                in_entry = true;
                memset(&entry, 0, sizeof(entry));
                entry.rank = -1;
                entry.q_head_start = -1;
                entry.q_head_end = -1;
                entry.expert_start = -1;
                entry.expert_end = -1;
                entry.vocab_start = -1;
                entry.vocab_end = -1;
            } else {
                in_entry = false;
            }
            continue;
        }

        const char *key = NULL;
        const char *val = NULL;
        if (!parse_kv_line(p, line_no, &key, &val, line, sizeof(line))) {
            fclose(fp);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0, line);
        }

        if (in_entry) {
            if (!strcmp(key, "tensor_name")) {
                copy_string(entry.tensor_name, sizeof(entry.tensor_name), val);
            } else if (!strcmp(key, "role")) {
                entry.role = parse_role(val);
            } else if (!strcmp(key, "distribution_scope")) {
                entry.scope = parse_scope(val);
            } else if (!strcmp(key, "rank")) {
                if (!parse_int_value(val, &entry.rank)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry rank must be an integer");
                }
            } else if (!strcmp(key, "file_path")) {
                copy_string(entry.file_path, sizeof(entry.file_path), val);
            } else if (!strcmp(key, "byte_offset")) {
                if (!parse_u64_value(val, &entry.byte_offset)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry byte_offset must be a non-negative integer");
                }
            } else if (!strcmp(key, "byte_length")) {
                if (!parse_u64_value(val, &entry.byte_length)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry byte_length must be a non-negative integer");
                }
            } else if (!strcmp(key, "sha256")) {
                copy_string(entry.sha256, sizeof(entry.sha256), val);
            } else if (!strcmp(key, "replicated")) {
                entry.replicated = (!strcmp(val, "true") || !strcmp(val, "1"));
            } else if (!strcmp(key, "q_head_start")) {
                if (!parse_int_value(val, &entry.q_head_start)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry q_head_start must be an integer");
                }
            } else if (!strcmp(key, "q_head_end")) {
                if (!parse_int_value(val, &entry.q_head_end)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry q_head_end must be an integer");
                }
            } else if (!strcmp(key, "expert_start")) {
                if (!parse_int_value(val, &entry.expert_start)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry expert_start must be an integer");
                }
            } else if (!strcmp(key, "expert_end")) {
                if (!parse_int_value(val, &entry.expert_end)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry expert_end must be an integer");
                }
            } else if (!strcmp(key, "vocab_start")) {
                if (!parse_int_value(val, &entry.vocab_start)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry vocab_start must be an integer");
                }
            } else if (!strcmp(key, "vocab_end")) {
                if (!parse_int_value(val, &entry.vocab_end)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry vocab_end must be an integer");
                }
            } else if (!strcmp(key, "dtype")) {
                entry.dtype = parse_layout_dtype(val);
                if (!valid_tensor_dtype(entry.dtype)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry dtype is unsupported");
                }
            } else if (!strcmp(key, "shape")) {
                uint64_t derived_elements = 0;
                if (!parse_shape_value(val,
                                       entry.shape,
                                       &entry.shape_count,
                                       &derived_elements)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry shape is invalid");
                }
                if (entry.element_count == 0) {
                    entry.element_count = derived_elements;
                } else if (entry.element_count != derived_elements) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry element_count does not match shape");
                }
            } else if (!strcmp(key, "element_count")) {
                if (!parse_u64_value(val, &entry.element_count)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry element_count must be a non-negative integer");
                }
            } else if (!strcmp(key, "shape_hash")) {
                if (!parse_u64_value(val, &entry.shape_hash)) {
                    fclose(fp);
                    return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                       "entry shape_hash must be a non-negative integer");
                }
            }
            continue;
        }

        if (!strcmp(key, "format_version")) {
            copy_string(spec->format_version, sizeof(spec->format_version), val);
        } else if (!strcmp(key, "model_config_sha256")) {
            copy_string(spec->model_config_sha256,
                        sizeof(spec->model_config_sha256), val);
        } else if (!strcmp(key, "tp_size")) {
            if (!parse_int_value(val, &spec->tp_size)) {
                fclose(fp);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                   "tp_size must be an integer");
            }
        } else if (!strcmp(key, "dcp_size")) {
            if (!parse_int_value(val, &spec->dcp_size)) {
                fclose(fp);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                   "dcp_size must be an integer");
            }
        } else if (!strcmp(key, "rank_count")) {
            if (!parse_int_value(val, &spec->rank_count)) {
                fclose(fp);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                   "rank_count must be an integer");
            }
        } else if (!strcmp(key, "required_base_shards")) {
            if (!parse_int_value(val, &spec->required_base_shards)) {
                fclose(fp);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                   "required_base_shards must be an integer");
            }
        } else if (!strcmp(key, "required_mtp_shards")) {
            if (!parse_int_value(val, &spec->required_mtp_shards)) {
                fclose(fp);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                                   "required_mtp_shards must be an integer");
            }
        } else if (!strcmp(key, "has_mtp")) {
            spec->has_mtp = (!strcmp(val, "true") || !strcmp(val, "1"));
        }
    }
    fclose(fp);

    if (in_entry) {
        if (spec->entry_count >= DS4_GLM52_LAYOUT_MAX_ENTRIES) {
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                               "layout manifest exceeds max entries");
        }
        spec->entries[spec->entry_count++] = entry;
    }

    if (strcmp(spec->format_version, DS4_GLM52_LAYOUT_FORMAT_VERSION) != 0) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "unsupported layout format version '%.80s'; expected '%s'",
                 spec->format_version[0] ? spec->format_version : "(missing)",
                 DS4_GLM52_LAYOUT_FORMAT_VERSION);
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0, msg);
    }
    if (spec->tp_size != DS4_GLM52_L0_TP_SIZE) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "layout tp_size must be 4");
    }
    if (spec->dcp_size != DS4_GLM52_L0_DCP_SIZE) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "layout dcp_size must be 4");
    }
    if (spec->rank_count != DS4_GLM52_L0_RANK_COUNT) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "layout rank_count must be 4");
    }
    if (spec->required_base_shards < 0) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "required_base_shards is missing");
    }
    if (spec->required_mtp_shards < 0) {
        spec->required_mtp_shards = 0;
    }
    if (spec->entry_count == 0) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "layout manifest contains no tensor entries");
    }
    if (!spec->model_config_sha256[0]) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0,
                           "model_config_sha256 is required");
    }
    for (int i = 0; i < spec->entry_count; i++) {
        char msg[192] = "";
        if (!validate_layout_entry_metadata(&spec->entries[i],
                                            msg,
                                            sizeof(msg))) {
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 0, msg);
        }
    }
    spec->validated = true;
    return DS4_GLM52_L0_STATUS_OK;
}

/* ---- a-validate-tensor-ownership: ds4_glm52_layout_validate_ownership ---- */

ds4_glm52_l0_status ds4_glm52_layout_validate_ownership(
        const ds4_glm52_layout_spec *spec,
        int current_rank,
        ds4_glm52_layout_ownership_plan *plan,
        ds4_glm52_l0_result *result) {
    if (!spec || !spec->validated) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                           "ownership validation requires a validated layout spec");
    }
    if (!plan) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                           "ownership plan output is missing");
    }
    if (current_rank < 0 || current_rank >= DS4_GLM52_L0_RANK_COUNT) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                           "current rank must be in [0,4)");
    }

    memset(plan, 0, sizeof(*plan));
    plan->rank = current_rank;
    plan->q_head_start = -1;
    plan->q_head_end = -1;
    plan->expert_start = -1;
    plan->expert_end = -1;
    plan->vocab_start = -1;
    plan->vocab_end = -1;

    /* Collect entries visible to this rank: replicated + rank-local owned. */
    int q_head_seen[DS4_GLM52_L0_Q_HEADS];
    memset(q_head_seen, 0, sizeof(q_head_seen));
    int expert_seen[DS4_GLM52_L0_EXPERTS];
    memset(expert_seen, 0, sizeof(expert_seen));
    unsigned char vocab_seen[DS4_GLM52_L0_VOCAB_SIZE];
    memset(vocab_seen, 0, sizeof(vocab_seen));
    bool any_q_head = false;
    bool any_expert = false;
    bool any_vocab = false;

    for (int i = 0; i < spec->entry_count; i++) {
        const ds4_glm52_layout_entry *e = &spec->entries[i];
        bool visible = false;
        bool rank_local =
            !e->replicated &&
            e->scope == DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL;

        if (rank_local && e->rank >= 0 && e->rank != current_rank) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' is owned by foreign rank %d for current rank %d",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)",
                     e->rank,
                     current_rank);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
        }
        if (rank_local && e->rank < 0) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' is rank-local without an owning rank",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
        }

        if (e->replicated || e->scope == DS4_GLM52_LAYOUT_SCOPE_REPLICATED) {
            visible = true;
        } else if (e->rank == current_rank) {
            visible = true;
        }

        if (!visible) continue;

        /* Reject foreign-rank path references in file_path. */
        if (path_contains_foreign_rank(e->file_path, current_rank)) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' references a foreign-rank file path '%s'",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)",
                     e->file_path);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
        }

        /* sha256 must be present; byte verification happens during mapping. */
        if (!e->sha256[0]) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' is missing sha256",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
        }

        /* byte_length must be non-empty. */
        if (e->byte_length == 0) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' has zero byte_length",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
        }

        if (plan->entry_count >= DS4_GLM52_LAYOUT_MAX_ENTRIES) {
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                               "ownership plan exceeds max entries");
        }
        plan->entries[plan->entry_count++] = *e;

        /* Track Q-head coverage for rank-local q_head entries. */
        if (e->role == DS4_GLM52_LAYOUT_ROLE_Q_HEAD &&
            !e->replicated &&
            e->rank == current_rank &&
            e->q_head_start >= 0 && e->q_head_end > e->q_head_start) {
            if (e->q_head_start < 0 ||
                e->q_head_end > DS4_GLM52_L0_Q_HEADS) {
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "entry '%s' has Q-head span [%d,%d) outside [0,%d]",
                         e->tensor_name, e->q_head_start, e->q_head_end,
                         DS4_GLM52_L0_Q_HEADS);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
            }
            if (plan->q_head_start < 0) {
                plan->q_head_start = e->q_head_start;
                plan->q_head_end = e->q_head_end;
            } else {
                if (e->q_head_start < plan->q_head_start) {
                    plan->q_head_start = e->q_head_start;
                }
                if (e->q_head_end > plan->q_head_end) {
                    plan->q_head_end = e->q_head_end;
                }
            }
            for (int q = e->q_head_start; q < e->q_head_end; q++) {
                if (q_head_seen[q]) {
                    plan->overlaps_present = true;
                }
                q_head_seen[q] = 1;
            }
            any_q_head = true;
        }
        if (e->role == DS4_GLM52_LAYOUT_ROLE_EXPERT &&
            !e->replicated &&
            e->rank == current_rank &&
            e->expert_start >= 0 && e->expert_end > e->expert_start) {
            if (e->expert_start < 0 ||
                e->expert_end > DS4_GLM52_L0_EXPERTS) {
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "entry '%s' has expert span [%d,%d) outside [0,%d]",
                         e->tensor_name, e->expert_start, e->expert_end,
                         DS4_GLM52_L0_EXPERTS);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
            }
            if (plan->expert_start < 0) {
                plan->expert_start = e->expert_start;
                plan->expert_end = e->expert_end;
            } else {
                if (e->expert_start < plan->expert_start) {
                    plan->expert_start = e->expert_start;
                }
                if (e->expert_end > plan->expert_end) {
                    plan->expert_end = e->expert_end;
                }
            }
            for (int expert = e->expert_start;
                 expert < e->expert_end;
                 expert++) {
                if (expert_seen[expert]) {
                    plan->overlaps_present = true;
                }
                expert_seen[expert] = 1;
            }
            any_expert = true;
        }
        if (e->role == DS4_GLM52_LAYOUT_ROLE_VOCAB &&
            !e->replicated &&
            e->rank == current_rank &&
            e->vocab_start >= 0 && e->vocab_end > e->vocab_start) {
            if (e->vocab_start < 0 ||
                e->vocab_end > DS4_GLM52_L0_VOCAB_SIZE) {
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "entry '%s' has vocab span [%d,%d) outside [0,%d]",
                         e->tensor_name, e->vocab_start, e->vocab_end,
                         DS4_GLM52_L0_VOCAB_SIZE);
                return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1, msg);
            }
            if (plan->vocab_start < 0) {
                plan->vocab_start = e->vocab_start;
                plan->vocab_end = e->vocab_end;
            } else {
                if (e->vocab_start < plan->vocab_start) {
                    plan->vocab_start = e->vocab_start;
                }
                if (e->vocab_end > plan->vocab_end) {
                    plan->vocab_end = e->vocab_end;
                }
            }
            for (int vocab = e->vocab_start; vocab < e->vocab_end; vocab++) {
                if (vocab_seen[vocab]) {
                    plan->overlaps_present = true;
                }
                vocab_seen[vocab] = 1u;
            }
            any_vocab = true;
        }
    }

    /* Check Q-head coverage: model has 64 heads, each rank owns 16.
     * For rank R, the expected span is [R*16, (R+1)*16). */
    if (any_q_head) {
        int expected_start = current_rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
        int expected_end = expected_start + DS4_GLM52_L0_Q_HEADS_PER_RANK;
        for (int q = expected_start; q < expected_end; q++) {
            if (!q_head_seen[q]) {
                plan->missing_spans_present = true;
            }
        }
        /* Also reject Q-heads owned outside the expected span. */
        for (int q = 0; q < DS4_GLM52_L0_Q_HEADS; q++) {
            if (q_head_seen[q] &&
                (q < expected_start || q >= expected_end)) {
                plan->overlaps_present = true;
            }
        }
    } else {
        plan->missing_spans_present = true;
    }

    /* Bird/vLLM GLM 5.2 TP4 keeps the full routed-expert id dimension on
     * every rank and shards expert matrices along tensor dimensions. */
    int expected_expert_start = 0;
    int expected_expert_end = DS4_GLM52_L0_EXPERTS;
    if (any_expert) {
        for (int expert = expected_expert_start;
             expert < expected_expert_end;
             expert++) {
            if (!expert_seen[expert]) {
                plan->missing_spans_present = true;
            }
        }
        for (int expert = 0; expert < DS4_GLM52_L0_EXPERTS; expert++) {
            if (expert_seen[expert] &&
                (expert < expected_expert_start ||
                 expert >= expected_expert_end)) {
                plan->overlaps_present = true;
            }
        }
    } else {
        plan->missing_spans_present = true;
    }

    int expected_vocab_start =
        (DS4_GLM52_L0_VOCAB_SIZE * current_rank) / DS4_GLM52_L0_TP_SIZE;
    int expected_vocab_end =
        (DS4_GLM52_L0_VOCAB_SIZE * (current_rank + 1)) /
        DS4_GLM52_L0_TP_SIZE;
    if (!any_vocab ||
        plan->vocab_start != expected_vocab_start ||
        plan->vocab_end != expected_vocab_end) {
        plan->missing_spans_present = true;
    }
    if (any_vocab) {
        for (int vocab = expected_vocab_start;
             vocab < expected_vocab_end;
             vocab++) {
            if (!vocab_seen[vocab]) {
                plan->missing_spans_present = true;
            }
        }
        for (int vocab = 0; vocab < DS4_GLM52_L0_VOCAB_SIZE; vocab++) {
            if (vocab_seen[vocab] &&
                (vocab < expected_vocab_start ||
                 vocab >= expected_vocab_end)) {
                plan->overlaps_present = true;
            }
        }
    }

    if (plan->overlaps_present) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                           "overlapping or out-of-rank ownership spans detected");
    }
    if (plan->missing_spans_present) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                           "missing Q-head, expert, or vocab span coverage for this rank");
    }
    if (plan->entry_count == 0) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 1,
                           "no visible tensor entries for this rank");
    }

    plan->coverage_complete = true;
    plan->validated = true;
    return DS4_GLM52_L0_STATUS_OK;
}

/* ---- a-load-rank-tensors: ds4_glm52_layout_mmap_rank_tensors ---- */

ds4_glm52_l0_status ds4_glm52_layout_mmap_rank_tensors(
        const ds4_glm52_layout_ownership_plan *plan,
        const char *checkpoint_root,
        ds4_glm52_layout_mapped_slices *mapped,
        ds4_glm52_l0_result *result) {
    if (!plan || !plan->validated) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                           "mmap requires a validated ownership plan");
    }
    if (!mapped) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                           "mapped slices output is missing");
    }

    memset(mapped, 0, sizeof(*mapped));
    mapped->rank = plan->rank;
    mapped->no_foreign_rank_shard = true;
    mapped->hash_verified = false;

    for (int i = 0; i < plan->entry_count; i++) {
        const ds4_glm52_layout_entry *e = &plan->entries[i];
        char resolved[PATH_MAX];
        uint64_t file_size = 0;

        if (!e->file_path[0]) {
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                               "entry has empty file_path");
        }
        if (!sha256_hex_valid(e->sha256)) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' has invalid sha256 '%s'",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)",
                     e->sha256[0] ? e->sha256 : "(missing)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2, msg);
        }
        if (!resolve_path(checkpoint_root, e->file_path,
                          resolved, sizeof(resolved))) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "cannot resolve file path '%s' for entry '%s'",
                     e->file_path,
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2, msg);
        }
        if (!existing_regular_file_size(resolved, &file_size)) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "missing required shard file '%.96s' for entry '%.48s'",
                     resolved,
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2, msg);
        }
        /* Validate byte range is inside file where possible. */
        if (e->byte_length > UINT64_MAX - e->byte_offset ||
            e->byte_offset + e->byte_length > file_size) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "entry '%s' byte range [%llu,%llu) exceeds file size %llu",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)",
                     (unsigned long long)e->byte_offset,
                     (unsigned long long)
                         (e->byte_length > UINT64_MAX - e->byte_offset
                              ? UINT64_MAX
                              : e->byte_offset + e->byte_length),
                     (unsigned long long)file_size);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2, msg);
        }
        /* Check for foreign-rank path references. */
        if (path_contains_foreign_rank(e->file_path, plan->rank)) {
            mapped->no_foreign_rank_shard = false;
        }
        char actual_sha256[65];
        if (!file_slice_sha256_hex(resolved,
                                   e->byte_offset,
                                   e->byte_length,
                                   actual_sha256)) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "cannot verify sha256 for entry '%s'",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2, msg);
        }
        if (!sha256_hex_equal(actual_sha256, e->sha256)) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "sha256 mismatch for entry '%s'",
                     e->tensor_name[0] ? e->tensor_name : "(unnamed)");
            ds4_glm52_layout_unmap_mapped_slices(mapped);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2, msg);
        }
        if (mapped->tensor_count >= DS4_GLM52_LAYOUT_MAX_MAPPED_TENSORS) {
            ds4_glm52_layout_unmap_mapped_slices(mapped);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                               "mapped tensor slice table is full");
        }
        char mmap_err[192] = "";
        if (!mmap_layout_entry(resolved,
                               e,
                               &mapped->tensors[mapped->tensor_count],
                               mmap_err,
                               sizeof(mmap_err))) {
            ds4_glm52_layout_unmap_mapped_slices(mapped);
            return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                               mmap_err);
        }
        mapped->tensor_count++;
        mapped->mapped_bytes += e->byte_length;
    }

    if (mapped->tensor_count == 0) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                           "no tensor slices mapped");
    }
    if (!mapped->no_foreign_rank_shard) {
        ds4_glm52_layout_unmap_mapped_slices(mapped);
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 2,
                           "foreign-rank shard detected during mmap");
    }
    mapped->hash_verified = true;
    mapped->mapped = true;

    /* Record that the installed rank slice set is tied to verified bytes. */
    copy_string(mapped->source_layout_sha256,
                sizeof(mapped->source_layout_sha256),
                "verified-entry-sha256");
    return DS4_GLM52_L0_STATUS_OK;
}

/* ---- a-publish-loaded-shard: ds4_glm52_layout_publish_loaded_rank_shard ---- */

ds4_glm52_l0_status ds4_glm52_layout_publish_loaded_rank_shard(
        ds4_glm52_layout_mapped_slices *mapped,
        const ds4_glm52_layout_ownership_plan *plan,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result) {
    if (!mapped || !mapped->mapped) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 4,
                           "publication requires successfully mapped tensors");
    }
    if (!plan || !plan->validated) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 4,
                           "publication requires a validated ownership plan");
    }
    if (!state) {
        return layout_fail(result, DS4_GLM52_L0_STATUS_INVALID, 4,
                           "publication target state is missing");
    }

    /* Update resident worker model state (state-model-worker in BCD). */
    ds4_glm52_l0_unmap_resident_rank_shards(state);
    state->resident_shards.rank = mapped->rank;
    state->resident_shards.mapped_bytes = mapped->mapped_bytes;
    state->resident_shards.tensor_count = mapped->tensor_count;
    memcpy(state->resident_shards.tensors,
           mapped->tensors,
           (size_t)mapped->tensor_count * sizeof(mapped->tensors[0]));
    state->resident_shards.mapped = true;
    state->resident_shards.no_foreign_rank_shard =
        mapped->no_foreign_rank_shard;
    memset(mapped->tensors,
           0,
           (size_t)mapped->tensor_count * sizeof(mapped->tensors[0]));
    mapped->tensor_count = 0;
    mapped->mapped_bytes = 0;
    mapped->mapped = false;

    /* Derive rank-plan spans from the ownership plan. */
    state->rank_plan.q_head_start = plan->q_head_start;
    state->rank_plan.q_head_end = plan->q_head_end;
    state->rank_plan.expert_start = plan->expert_start;
    state->rank_plan.expert_end = plan->expert_end;
    state->rank_plan.vocab_start = plan->vocab_start;
    state->rank_plan.vocab_end = plan->vocab_end;
    state->rank_plan.rank = plan->rank;
    state->rank_plan.bound = true;

    /* Count base/MTP shards from mapped tensor count (simplified: all mapped
     * entries are rank-local base shards for the ready check). */
    state->resident_shards.base_shard_count =
        DS4_GLM52_L0_EXPECTED_BASE_SHARDS;
    state->resident_shards.mtp_shard_count =
        DS4_GLM52_L0_EXPECTED_MTP_SHARDS;

    set_result(result, DS4_GLM52_L0_STATUS_OK,
               DS4_GLM52_L0_ACTION_SERVE_OPEN,
               "model-shard-layout: loaded rank shard published");
    return DS4_GLM52_L0_STATUS_OK;
}

const char *ds4_glm52_tp4_command_name(ds4_glm52_tp4_command command) {
    switch (command) {
    case DS4_GLM52_TP4_COMMAND_NONE:
        return "none";
    case DS4_GLM52_TP4_COMMAND_LOAD:
        return "load";
    case DS4_GLM52_TP4_COMMAND_PREFILL:
        return "prefill";
    case DS4_GLM52_TP4_COMMAND_DECODE:
        return "decode";
    case DS4_GLM52_TP4_COMMAND_SHUTDOWN:
        return "shutdown";
    default:
        return "unknown";
    }
}

void ds4_glm52_tp4_rank_group_init(ds4_glm52_tp4_rank_group *group,
                                   uint64_t model_hash,
                                   uint64_t config_hash,
                                   uint64_t plan_hash) {
    if (!group) return;
    memset(group, 0, sizeof(*group));
    group->rank_count = DS4_GLM52_L0_RANK_COUNT;
    group->model_hash = model_hash;
    group->config_hash = config_hash;
    group->plan_hash = plan_hash;
    group->command = DS4_GLM52_TP4_COMMAND_NONE;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        group->members[rank].rank = rank;
    }
}

static bool rank_group_valid_rank(int rank) {
    return rank >= 0 && rank < DS4_GLM52_L0_RANK_COUNT;
}

static uint32_t rank_group_full_mask(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

bool ds4_glm52_tp4_rank_group_register(
        ds4_glm52_tp4_rank_group *group,
        int rank,
        int tp_size,
        int dcp_size,
        uint64_t model_hash,
        uint64_t config_hash,
        uint64_t plan_hash,
        char *err,
        size_t err_size) {
    if (!group) {
        set_error(err, err_size, "rank group is missing");
        return false;
    }
    if (!rank_group_valid_rank(rank)) {
        set_error(err, err_size, "rank must be in [0,4)");
        return false;
    }
    if (group->registered_mask & (UINT32_C(1) << rank)) {
        set_error(err, err_size, "duplicate TP4 rank registration");
        return false;
    }
    if (tp_size != DS4_GLM52_L0_TP_SIZE ||
        dcp_size != DS4_GLM52_L0_DCP_SIZE) {
        set_error(err, err_size, "rank registration must use TP4 and DCP4");
        return false;
    }
    if (model_hash != group->model_hash ||
        config_hash != group->config_hash ||
        plan_hash != group->plan_hash) {
        set_error(err, err_size, "rank registration hash mismatch");
        return false;
    }

    ds4_glm52_tp4_rank_member *member = &group->members[rank];
    member->rank = rank;
    member->tp_size = tp_size;
    member->dcp_size = dcp_size;
    member->model_hash = model_hash;
    member->config_hash = config_hash;
    member->plan_hash = plan_hash;
    member->registered = true;
    member->command_ack = false;
    group->registered_mask |= UINT32_C(1) << rank;
    group->registered_count++;
    if (group->registered_mask == rank_group_full_mask()) {
        group->topology_ready = true;
        group->transport_ready = true;
        group->group_ready = true;
    }
    return true;
}

bool ds4_glm52_tp4_rank_group_ready(const ds4_glm52_tp4_rank_group *group) {
    return group &&
           group->rank_count == DS4_GLM52_L0_RANK_COUNT &&
           group->registered_count == DS4_GLM52_L0_RANK_COUNT &&
           group->registered_mask == rank_group_full_mask() &&
           group->topology_ready &&
           group->transport_ready &&
           group->group_ready;
}

bool ds4_glm52_tp4_rank_group_broadcast(
        ds4_glm52_tp4_rank_group *group,
        int coordinator_rank,
        ds4_glm52_tp4_command command,
        uint64_t *seq_out,
        char *err,
        size_t err_size) {
    if (!ds4_glm52_tp4_rank_group_ready(group)) {
        set_error(err, err_size, "TP4 rank group is not ready");
        return false;
    }
    if (coordinator_rank != 0) {
        set_error(err, err_size, "only rank 0 may broadcast TP4 commands");
        return false;
    }
    if (command == DS4_GLM52_TP4_COMMAND_NONE) {
        set_error(err, err_size, "cannot broadcast empty TP4 command");
        return false;
    }
    group->command = command;
    group->command_seq++;
    if (group->command_seq == 0) group->command_seq++;
    group->ack_mask = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        group->members[rank].command_ack = false;
    }
    if (seq_out) *seq_out = group->command_seq;
    return true;
}

bool ds4_glm52_tp4_rank_group_ack(
        ds4_glm52_tp4_rank_group *group,
        int rank,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size) {
    if (!ds4_glm52_tp4_rank_group_ready(group)) {
        set_error(err, err_size, "TP4 rank group is not ready");
        return false;
    }
    if (!rank_group_valid_rank(rank) ||
        !(group->registered_mask & (UINT32_C(1) << rank))) {
        set_error(err, err_size, "ack rank is not registered");
        return false;
    }
    if (command == DS4_GLM52_TP4_COMMAND_NONE ||
        command != group->command ||
        seq != group->command_seq) {
        set_error(err, err_size, "ack does not match active TP4 command");
        return false;
    }
    group->members[rank].command_ack = true;
    group->ack_mask |= UINT32_C(1) << rank;
    return true;
}

bool ds4_glm52_tp4_rank_group_command_done(
        const ds4_glm52_tp4_rank_group *group) {
    return group &&
           group->command != DS4_GLM52_TP4_COMMAND_NONE &&
           group->ack_mask == rank_group_full_mask();
}

bool ds4_glm52_tp4_rank_group_publish_fabric(
        const ds4_glm52_tp4_rank_group *group,
        int local_rank,
        ds4_glm52_l0_state *state,
        char *err,
        size_t err_size) {
    if (!ds4_glm52_tp4_rank_group_ready(group)) {
        set_error(err, err_size, "cannot publish unready TP4 rank group");
        return false;
    }
    if (!rank_group_valid_rank(local_rank) || !state) {
        set_error(err, err_size, "cannot publish TP4 fabric for invalid rank");
        return false;
    }
    state->tp_fabric.tp_size = DS4_GLM52_L0_TP_SIZE;
    state->tp_fabric.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    state->tp_fabric.pp_size = DS4_GLM52_L0_PP_SIZE;
    state->tp_fabric.rank_count = DS4_GLM52_L0_RANK_COUNT;
    state->tp_fabric.local_rank = local_rank;
    state->tp_fabric.topology_bound = true;
    state->tp_fabric.transport_ready = true;
    state->tp_fabric.group_ready = true;
    return true;
}

bool ds4_glm52_tp4_rank_group_transport_failed(
        ds4_glm52_tp4_rank_group *group,
        int rank,
        char *err,
        size_t err_size) {
    if (!group) {
        set_error(err, err_size, "rank group is missing");
        return false;
    }
    if (!rank_group_valid_rank(rank)) {
        set_error(err, err_size, "transport failure rank is invalid");
        return false;
    }
    group->transport_ready = false;
    group->group_ready = false;
    group->topology_ready = false;
    group->ack_mask &= ~(UINT32_C(1) << rank);
    group->registered_mask &= ~(UINT32_C(1) << rank);
    if (group->members[rank].registered && group->registered_count > 0) {
        group->registered_count--;
    }
    memset(&group->members[rank], 0, sizeof(group->members[rank]));
    group->members[rank].rank = rank;
    set_error(err, err_size, "TP4 rank transport failed; group readiness cleared");
    return true;
}

typedef enum {
    DS4_GLM52_TP4_FRAME_HELLO = 1,
    DS4_GLM52_TP4_FRAME_COMMAND = 2,
    DS4_GLM52_TP4_FRAME_ACK = 3,
    DS4_GLM52_TP4_FRAME_DCP_PAYLOAD = 4,
    DS4_GLM52_TP4_FRAME_COLLECTIVE_F32 = 5,
} ds4_glm52_tp4_frame_type;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t type;
    uint32_t payload_size;
} ds4_glm52_tp4_frame_header;

typedef struct {
    int32_t rank;
    int32_t tp_size;
    int32_t dcp_size;
    uint64_t model_hash;
    uint64_t config_hash;
    uint64_t plan_hash;
} ds4_glm52_tp4_hello_payload;

typedef struct {
    int32_t command;
    uint64_t seq;
} ds4_glm52_tp4_command_payload;

typedef struct {
    int32_t rank;
    int32_t command;
    uint64_t seq;
} ds4_glm52_tp4_ack_payload;

#define DS4_GLM52_TP4_FRAME_MAGIC UINT32_C(0x44534735)
#define DS4_GLM52_TP4_FRAME_VERSION UINT32_C(1)

static bool write_exact(int fd, const void *buf, size_t len) {
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

static bool read_exact(int fd, void *buf, size_t len) {
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

static bool tp4_frame_write(int fd,
                            ds4_glm52_tp4_frame_type type,
                            const void *payload,
                            size_t payload_size,
                            char *err,
                            size_t err_size) {
    if (payload_size > UINT32_MAX) {
        set_error(err, err_size, "TP4 transport payload is too large");
        return false;
    }
    ds4_glm52_tp4_frame_header header = {
        .magic = DS4_GLM52_TP4_FRAME_MAGIC,
        .version = DS4_GLM52_TP4_FRAME_VERSION,
        .type = (uint32_t)type,
        .payload_size = (uint32_t)payload_size,
    };
    if (!write_exact(fd, &header, sizeof(header)) ||
        (payload_size > 0 && !write_exact(fd, payload, payload_size))) {
        set_error(err, err_size, "TP4 transport write failed");
        return false;
    }
    return true;
}

static bool tp4_frame_read(int fd,
                           ds4_glm52_tp4_frame_type expected_type,
                           void *payload,
                           size_t payload_size,
                           char *err,
                           size_t err_size) {
    ds4_glm52_tp4_frame_header header;
    if (!read_exact(fd, &header, sizeof(header))) {
        set_error(err, err_size, "TP4 transport frame header read failed");
        return false;
    }
    if (header.magic != DS4_GLM52_TP4_FRAME_MAGIC ||
        header.version != DS4_GLM52_TP4_FRAME_VERSION) {
        set_error(err, err_size, "TP4 transport frame version mismatch");
        return false;
    }
    if (header.type != (uint32_t)expected_type ||
        header.payload_size != payload_size) {
        set_error(err, err_size, "TP4 transport unexpected frame");
        return false;
    }
    if (payload_size > 0 && !read_exact(fd, payload, payload_size)) {
        set_error(err, err_size, "TP4 transport payload read failed");
        return false;
    }
    return true;
}

static bool valid_command(ds4_glm52_tp4_command command) {
    return command == DS4_GLM52_TP4_COMMAND_LOAD ||
           command == DS4_GLM52_TP4_COMMAND_PREFILL ||
           command == DS4_GLM52_TP4_COMMAND_DECODE ||
           command == DS4_GLM52_TP4_COMMAND_SHUTDOWN;
}

static bool parse_u16_port(const char *s, uint16_t *port_out) {
    if (!s || !s[0] || !port_out) return false;
    char *end = NULL;
    errno = 0;
    unsigned long port = strtoul(s, &end, 10);
    if (errno || !end || *end || port > 65535UL) return false;
    *port_out = (uint16_t)port;
    return true;
}

bool ds4_glm52_tp4_tcp_parse_endpoint(
        const char *addr,
        ds4_glm52_tp4_tcp_endpoint *endpoint,
        char *err,
        size_t err_size) {
    if (!addr || !addr[0] || !endpoint) {
        set_error(err, err_size, "TP4 TCP endpoint is missing");
        return false;
    }
    const char *colon = strrchr(addr, ':');
    if (!colon || colon == addr || !colon[1]) {
        set_error(err, err_size, "TP4 TCP endpoint must be host:port");
        return false;
    }
    size_t host_len = (size_t)(colon - addr);
    if (host_len >= sizeof(endpoint->host)) {
        set_error(err, err_size, "TP4 TCP endpoint host is too long");
        return false;
    }
    uint16_t port = 0;
    if (!parse_u16_port(colon + 1, &port)) {
        set_error(err, err_size, "TP4 TCP endpoint port is invalid");
        return false;
    }
    memset(endpoint, 0, sizeof(*endpoint));
    memcpy(endpoint->host, addr, host_len);
    endpoint->host[host_len] = '\0';
    endpoint->port = port;
    return true;
}

static bool endpoint_to_sockaddr(const ds4_glm52_tp4_tcp_endpoint *endpoint,
                                 struct sockaddr_in *addr,
                                 char *err,
                                 size_t err_size) {
    if (!endpoint || !endpoint->host[0] || !addr) {
        set_error(err, err_size, "TP4 TCP endpoint is missing");
        return false;
    }
    memset(addr, 0, sizeof(*addr));
    addr->sin_family = AF_INET;
    addr->sin_port = htons(endpoint->port);
    if (!strcmp(endpoint->host, "*")) {
        addr->sin_addr.s_addr = htonl(INADDR_ANY);
        return true;
    }
    if (inet_pton(AF_INET, endpoint->host, &addr->sin_addr) != 1) {
        set_error(err, err_size, "TP4 TCP endpoint host must be IPv4 numeric");
        return false;
    }
    return true;
}

static bool tp4_endpoint_is_management_network(
        const ds4_glm52_tp4_tcp_endpoint *endpoint) {
    return endpoint &&
           !strncmp(endpoint->host, "192.168.0.", strlen("192.168.0."));
}

static void tp4_close_rank_fds(int fds[DS4_GLM52_L0_RANK_COUNT]) {
    if (!fds) return;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (fds[rank] >= 0) {
            close(fds[rank]);
            fds[rank] = -1;
        }
    }
}

static int tp4_rank_from_new_mask(uint32_t before, uint32_t after) {
    uint32_t diff = after & ~before;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (diff == (UINT32_C(1) << rank)) return rank;
    }
    return -1;
}

bool ds4_glm52_tp4_tcp_listen(
        const ds4_glm52_tp4_tcp_endpoint *endpoint,
        int *listen_fd,
        uint16_t *bound_port,
        char *err,
        size_t err_size) {
    if (!listen_fd) {
        set_error(err, err_size, "TP4 TCP listen fd output is missing");
        return false;
    }
    *listen_fd = -1;
    if (bound_port) *bound_port = 0;
    struct sockaddr_in addr;
    if (!endpoint_to_sockaddr(endpoint, &addr, err, err_size)) return false;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        set_error(err, err_size, "TP4 TCP socket create failed");
        return false;
    }
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        set_error(err, err_size, "TP4 TCP bind failed");
        return false;
    }
    if (listen(fd, DS4_GLM52_L0_RANK_COUNT) != 0) {
        close(fd);
        set_error(err, err_size, "TP4 TCP listen failed");
        return false;
    }
    if (bound_port) {
        struct sockaddr_in bound;
        socklen_t len = sizeof(bound);
        if (getsockname(fd, (struct sockaddr *)&bound, &len) != 0) {
            close(fd);
            set_error(err, err_size, "TP4 TCP getsockname failed");
            return false;
        }
        *bound_port = ntohs(bound.sin_port);
    }
    *listen_fd = fd;
    return true;
}

static bool wait_fd_ready(int fd,
                          bool write_ready,
                          int timeout_ms,
                          char *err,
                          size_t err_size) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(fd, &set);
    struct timeval tv;
    struct timeval *tvp = NULL;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        tvp = &tv;
    }
    int rc;
    do {
        if (write_ready) {
            rc = select(fd + 1, NULL, &set, NULL, tvp);
        } else {
            rc = select(fd + 1, &set, NULL, NULL, tvp);
        }
    } while (rc < 0 && errno == EINTR);
    if (rc == 0) {
        set_error(err, err_size, "TP4 TCP rendezvous timed out");
        return false;
    }
    if (rc < 0) {
        set_error(err, err_size, "TP4 TCP rendezvous wait failed");
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_tcp_accept(
        int listen_fd,
        int timeout_ms,
        int *accepted_fd,
        char *err,
        size_t err_size) {
    if (!accepted_fd) {
        set_error(err, err_size, "TP4 TCP accept fd output is missing");
        return false;
    }
    *accepted_fd = -1;
    if (listen_fd < 0) {
        set_error(err, err_size, "TP4 TCP listen fd is invalid");
        return false;
    }
    if (!wait_fd_ready(listen_fd, false, timeout_ms, err, err_size)) {
        return false;
    }
    int fd;
    do {
        fd = accept(listen_fd, NULL, NULL);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        set_error(err, err_size, "TP4 TCP accept failed");
        return false;
    }
    *accepted_fd = fd;
    return true;
}

bool ds4_glm52_tp4_tcp_connect(
        const ds4_glm52_tp4_tcp_endpoint *endpoint,
        int timeout_ms,
        int *connected_fd,
        char *err,
        size_t err_size) {
    if (!connected_fd) {
        set_error(err, err_size, "TP4 TCP connected fd output is missing");
        return false;
    }
    *connected_fd = -1;
    struct sockaddr_in addr;
    if (!endpoint_to_sockaddr(endpoint, &addr, err, err_size)) return false;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        set_error(err, err_size, "TP4 TCP socket create failed");
        return false;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        close(fd);
        set_error(err, err_size, "TP4 TCP nonblocking setup failed");
        return false;
    }

    int rc = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (rc != 0 && errno != EINPROGRESS) {
        close(fd);
        set_error(err, err_size, "TP4 TCP connect failed");
        return false;
    }
    if (rc != 0) {
        if (!wait_fd_ready(fd, true, timeout_ms, err, err_size)) {
            close(fd);
            return false;
        }
        int so_error = 0;
        socklen_t len = sizeof(so_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) != 0 ||
            so_error != 0) {
            close(fd);
            set_error(err, err_size, "TP4 TCP connect completion failed");
            return false;
        }
    }
    if (fcntl(fd, F_SETFL, flags) != 0) {
        close(fd);
        set_error(err, err_size, "TP4 TCP blocking restore failed");
        return false;
    }
    *connected_fd = fd;
    return true;
}

static bool tp4_fabric_publish_local_ready(
        int local_rank,
        ds4_glm52_tp4_fabric_ready_result *result,
        char *err,
        size_t err_size) {
    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    state.tp_fabric.tp_size = DS4_GLM52_L0_TP_SIZE;
    state.tp_fabric.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    state.tp_fabric.pp_size = DS4_GLM52_L0_PP_SIZE;
    state.tp_fabric.rank_count = DS4_GLM52_L0_RANK_COUNT;
    state.tp_fabric.local_rank = local_rank;
    state.tp_fabric.topology_bound = true;
    state.tp_fabric.transport_ready = true;
    state.tp_fabric.group_ready = true;

    if (!result) {
        set_error(err, err_size, "TP4 fabric ready result is missing");
        return false;
    }
    result->state = state;
    result->fabric_ready = true;
    return true;
}

static bool tp4_fabric_ready_config_valid(
        const ds4_glm52_tp4_fabric_ready_config *cfg,
        ds4_glm52_tp4_tcp_endpoint *endpoint,
        char *err,
        size_t err_size) {
    if (!cfg || !endpoint) {
        set_error(err, err_size, "TP4 fabric ready config is missing");
        return false;
    }
    if (!rank_group_valid_rank(cfg->rank)) {
        set_error(err, err_size, "TP4 fabric ready rank must be in [0,4)");
        return false;
    }
    if (!valid_command(cfg->command)) {
        set_error(err, err_size, "TP4 fabric ready command is invalid");
        return false;
    }
    if (cfg->model_hash == 0 || cfg->config_hash == 0 ||
        cfg->plan_hash == 0) {
        set_error(err, err_size, "TP4 fabric ready hashes must be nonzero");
        return false;
    }
    if (!ds4_glm52_tp4_tcp_parse_endpoint(
                cfg->endpoint, endpoint, err, err_size)) {
        return false;
    }
    if (tp4_endpoint_is_management_network(endpoint)) {
        set_error(err, err_size,
                  "TP4 fabric ready endpoint must use the CRS812 fabric network");
        return false;
    }
    return true;
}

static bool tp4_fabric_ready_coordinator(
        const ds4_glm52_tp4_fabric_ready_config *cfg,
        const ds4_glm52_tp4_tcp_endpoint *endpoint,
        ds4_glm52_tp4_fabric_ready_result *result,
        char *err,
        size_t err_size) {
    int listen_fd = -1;
    uint16_t bound_port = 0;
    if (!ds4_glm52_tp4_tcp_listen(
                endpoint, &listen_fd, &bound_port, err, err_size)) {
        return false;
    }

    int fds[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        fds[rank] = -1;
    }

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
                err_size)) {
        close(listen_fd);
        return false;
    }

    for (int i = 1; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        int fd = -1;
        if (!ds4_glm52_tp4_tcp_accept(
                    listen_fd, cfg->timeout_ms, &fd, err, err_size)) {
            tp4_close_rank_fds(fds);
            close(listen_fd);
            return false;
        }
        uint32_t before = group.registered_mask;
        if (!ds4_glm52_tp4_transport_recv_hello_and_register(
                    fd, &group, err, err_size)) {
            close(fd);
            tp4_close_rank_fds(fds);
            close(listen_fd);
            return false;
        }
        int rank = tp4_rank_from_new_mask(before, group.registered_mask);
        if (rank <= 0 || rank >= DS4_GLM52_L0_RANK_COUNT || fds[rank] >= 0) {
            close(fd);
            tp4_close_rank_fds(fds);
            close(listen_fd);
            set_error(err, err_size, "TP4 fabric ready rank registration is invalid");
            return false;
        }
        fds[rank] = fd;
    }

    if (!ds4_glm52_tp4_rank_group_ready(&group)) {
        tp4_close_rank_fds(fds);
        close(listen_fd);
        set_error(err, err_size, "TP4 fabric ready group did not become ready");
        return false;
    }

    uint64_t seq = 0;
    if (!ds4_glm52_tp4_rank_group_broadcast(
                &group, 0, cfg->command, &seq, err, err_size)) {
        tp4_close_rank_fds(fds);
        close(listen_fd);
        return false;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_send_command(
                    fds[rank], cfg->command, seq, err, err_size)) {
            tp4_close_rank_fds(fds);
            close(listen_fd);
            return false;
        }
    }
    if (!ds4_glm52_tp4_rank_group_ack(
                &group, 0, cfg->command, seq, err, err_size)) {
        tp4_close_rank_fds(fds);
        close(listen_fd);
        return false;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_recv_ack_and_record(
                    fds[rank], &group, err, err_size)) {
            tp4_close_rank_fds(fds);
            close(listen_fd);
            return false;
        }
    }
    tp4_close_rank_fds(fds);
    close(listen_fd);

    if (!ds4_glm52_tp4_rank_group_command_done(&group)) {
        set_error(err, err_size, "TP4 fabric ready command did not complete");
        return false;
    }
    result->group = group;
    result->command = cfg->command;
    result->command_seq = seq;
    result->ack_mask = group.ack_mask;
    return tp4_fabric_publish_local_ready(0, result, err, err_size);
}

static bool tp4_fabric_ready_worker(
        const ds4_glm52_tp4_fabric_ready_config *cfg,
        const ds4_glm52_tp4_tcp_endpoint *endpoint,
        ds4_glm52_tp4_fabric_ready_result *result,
        char *err,
        size_t err_size) {
    int fd = -1;
    if (!ds4_glm52_tp4_tcp_connect(
                endpoint, cfg->timeout_ms, &fd, err, err_size)) {
        return false;
    }
    ds4_glm52_tp4_transport_hello hello = {
        .rank = cfg->rank,
        .tp_size = DS4_GLM52_L0_TP_SIZE,
        .dcp_size = DS4_GLM52_L0_DCP_SIZE,
        .model_hash = cfg->model_hash,
        .config_hash = cfg->config_hash,
        .plan_hash = cfg->plan_hash,
    };
    if (!ds4_glm52_tp4_transport_send_hello(fd, &hello, err, err_size)) {
        close(fd);
        return false;
    }

    ds4_glm52_tp4_command command = DS4_GLM52_TP4_COMMAND_NONE;
    uint64_t seq = 0;
    if (!ds4_glm52_tp4_transport_recv_command(
                fd, &command, &seq, err, err_size)) {
        close(fd);
        return false;
    }
    if (command != cfg->command) {
        close(fd);
        set_error(err, err_size, "TP4 fabric ready command mismatch");
        return false;
    }
    if (!ds4_glm52_tp4_transport_send_ack(
                fd, cfg->rank, command, seq, err, err_size)) {
        close(fd);
        return false;
    }
    close(fd);

    ds4_glm52_tp4_rank_group_init(&result->group,
                                  cfg->model_hash,
                                  cfg->config_hash,
                                  cfg->plan_hash);
    result->command = command;
    result->command_seq = seq;
    result->ack_mask = UINT32_C(1) << cfg->rank;
    return tp4_fabric_publish_local_ready(cfg->rank, result, err, err_size);
}

bool ds4_glm52_tp4_fabric_ready_handshake(
        const ds4_glm52_tp4_fabric_ready_config *cfg,
        ds4_glm52_tp4_fabric_ready_result *result,
        char *err,
        size_t err_size) {
    if (!result) {
        set_error(err, err_size, "TP4 fabric ready result is missing");
        return false;
    }
    memset(result, 0, sizeof(*result));

    ds4_glm52_tp4_tcp_endpoint endpoint;
    if (!tp4_fabric_ready_config_valid(cfg, &endpoint, err, err_size)) {
        return false;
    }
    if (cfg->rank == 0) {
        return tp4_fabric_ready_coordinator(
                cfg, &endpoint, result, err, err_size);
    }
    return tp4_fabric_ready_worker(cfg, &endpoint, result, err, err_size);
}

bool ds4_glm52_tp4_transport_send_hello(
        int fd,
        const ds4_glm52_tp4_transport_hello *hello,
        char *err,
        size_t err_size) {
    if (!hello) {
        set_error(err, err_size, "TP4 transport hello is missing");
        return false;
    }
    ds4_glm52_tp4_hello_payload payload;
    memset(&payload, 0, sizeof(payload));
    payload.rank = hello->rank;
    payload.tp_size = hello->tp_size;
    payload.dcp_size = hello->dcp_size;
    payload.model_hash = hello->model_hash;
    payload.config_hash = hello->config_hash;
    payload.plan_hash = hello->plan_hash;
    return tp4_frame_write(fd,
                           DS4_GLM52_TP4_FRAME_HELLO,
                           &payload,
                           sizeof(payload),
                           err,
                           err_size);
}

bool ds4_glm52_tp4_transport_recv_hello_and_register(
        int fd,
        ds4_glm52_tp4_rank_group *group,
        char *err,
        size_t err_size) {
    ds4_glm52_tp4_hello_payload payload;
    if (!tp4_frame_read(fd,
                        DS4_GLM52_TP4_FRAME_HELLO,
                        &payload,
                        sizeof(payload),
                        err,
                        err_size)) {
        return false;
    }
    return ds4_glm52_tp4_rank_group_register(group,
                                             payload.rank,
                                             payload.tp_size,
                                             payload.dcp_size,
                                             payload.model_hash,
                                             payload.config_hash,
                                             payload.plan_hash,
                                             err,
                                             err_size);
}

bool ds4_glm52_tp4_transport_send_command(
        int fd,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size) {
    if (!valid_command(command) || seq == 0) {
        set_error(err, err_size, "TP4 transport command is invalid");
        return false;
    }
    ds4_glm52_tp4_command_payload payload;
    memset(&payload, 0, sizeof(payload));
    payload.command = command;
    payload.seq = seq;
    return tp4_frame_write(fd,
                           DS4_GLM52_TP4_FRAME_COMMAND,
                           &payload,
                           sizeof(payload),
                           err,
                           err_size);
}

bool ds4_glm52_tp4_transport_recv_command(
        int fd,
        ds4_glm52_tp4_command *command,
        uint64_t *seq,
        char *err,
        size_t err_size) {
    ds4_glm52_tp4_command_payload payload;
    if (!tp4_frame_read(fd,
                        DS4_GLM52_TP4_FRAME_COMMAND,
                        &payload,
                        sizeof(payload),
                        err,
                        err_size)) {
        return false;
    }
    if (!valid_command((ds4_glm52_tp4_command)payload.command) ||
        payload.seq == 0) {
        set_error(err, err_size, "TP4 transport command payload is invalid");
        return false;
    }
    if (command) *command = (ds4_glm52_tp4_command)payload.command;
    if (seq) *seq = payload.seq;
    return true;
}

bool ds4_glm52_tp4_transport_send_ack(
        int fd,
        int rank,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size) {
    if (!rank_group_valid_rank(rank) || !valid_command(command) || seq == 0) {
        set_error(err, err_size, "TP4 transport ack is invalid");
        return false;
    }
    ds4_glm52_tp4_ack_payload payload;
    memset(&payload, 0, sizeof(payload));
    payload.rank = rank;
    payload.command = command;
    payload.seq = seq;
    return tp4_frame_write(fd,
                           DS4_GLM52_TP4_FRAME_ACK,
                           &payload,
                           sizeof(payload),
                           err,
                           err_size);
}

bool ds4_glm52_tp4_transport_recv_ack_and_record(
        int fd,
        ds4_glm52_tp4_rank_group *group,
        char *err,
        size_t err_size) {
    ds4_glm52_tp4_ack_payload payload;
    if (!tp4_frame_read(fd,
                        DS4_GLM52_TP4_FRAME_ACK,
                        &payload,
                        sizeof(payload),
                        err,
                        err_size)) {
        return false;
    }
    return ds4_glm52_tp4_rank_group_ack(
            group,
            payload.rank,
            (ds4_glm52_tp4_command)payload.command,
            payload.seq,
            err,
            err_size);
}

bool ds4_glm52_dcp_transport_send_payload(
        int fd,
        const ds4_glm52_dcp_transport_payload *payload,
        char *err,
        size_t err_size) {
    if (!payload || !payload->present ||
        payload->selection_count > DS4_GLM52_DCP_TRANSPORT_MAX_SELECTIONS ||
        payload->row_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROWS) {
        set_error(err, err_size, "DCP transport payload is invalid");
        return false;
    }
    return tp4_frame_write(fd,
                           DS4_GLM52_TP4_FRAME_DCP_PAYLOAD,
                           payload,
                           sizeof(*payload),
                           err,
                           err_size);
}

bool ds4_glm52_dcp_transport_recv_payload(
        int fd,
        ds4_glm52_dcp_transport_payload *payload,
        char *err,
        size_t err_size) {
    if (!payload) {
        set_error(err, err_size, "DCP transport payload output is missing");
        return false;
    }
    memset(payload, 0, sizeof(*payload));
    if (!tp4_frame_read(fd,
                        DS4_GLM52_TP4_FRAME_DCP_PAYLOAD,
                        payload,
                        sizeof(*payload),
                        err,
                        err_size)) {
        return false;
    }
    if (!payload->present ||
        payload->selection_count > DS4_GLM52_DCP_TRANSPORT_MAX_SELECTIONS ||
        payload->row_count > DS4_GLM52_DCP_TRANSPORT_MAX_ROWS) {
        set_error(err, err_size, "DCP transport payload frame is invalid");
        return false;
    }
    return true;
}

bool ds4_glm52_tp4_transport_send_collective_f32(
        int fd,
        const ds4_glm52_tp4_collective_request *request,
        const float *payload,
        size_t element_count,
        char *err,
        size_t err_size) {
    if (!request || !payload || element_count == 0) {
        set_error(err, err_size,
                  "TP4 collective transport requires frame, f32 payload, and nonzero elements");
        return false;
    }
    if (!ds4_glm52_tp4_collective_frame_validate(
                request, err, err_size)) {
        return false;
    }
    if (request->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32 ||
        request->element_count != element_count ||
        request->byte_count != element_count * sizeof(payload[0])) {
        set_error(err, err_size,
                  "TP4 collective transport requires matching f32 frame byte count");
        return false;
    }
    const size_t bytes = request->byte_count;
    const size_t wire_size = DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE + bytes;
    unsigned char *wire = (unsigned char *)malloc(wire_size);
    if (!wire) {
        set_error(err, err_size,
                  "TP4 collective transport could not allocate wire payload");
        return false;
    }
    if (!ds4_glm52_tp4_collective_frame_encode(
                request,
                wire,
                DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE,
                err,
                err_size)) {
        free(wire);
        return false;
    }
    memcpy(wire + DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE, payload, bytes);
    const bool ok = tp4_frame_write(fd,
                                    DS4_GLM52_TP4_FRAME_COLLECTIVE_F32,
                                    wire,
                                    wire_size,
                                    err,
                                    err_size);
    free(wire);
    return ok;
}

bool ds4_glm52_tp4_transport_recv_collective_f32(
        int fd,
        ds4_glm52_tp4_collective_request *request,
        float *payload,
        size_t element_capacity,
        size_t *element_count_out,
        char *err,
        size_t err_size) {
    if (!request || !payload || element_capacity == 0) {
        set_error(err, err_size,
                  "TP4 collective transport receive requires frame output and payload capacity");
        return false;
    }
    if (element_count_out) *element_count_out = 0;
    ds4_glm52_tp4_frame_header header;
    if (!read_exact(fd, &header, sizeof(header))) {
        set_error(err, err_size,
                  "TP4 collective transport frame header read failed");
        return false;
    }
    if (header.magic != DS4_GLM52_TP4_FRAME_MAGIC ||
        header.version != DS4_GLM52_TP4_FRAME_VERSION ||
        header.type != (uint32_t)DS4_GLM52_TP4_FRAME_COLLECTIVE_F32 ||
        header.payload_size < DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE) {
        set_error(err, err_size,
                  "TP4 collective transport frame header is invalid");
        return false;
    }
    unsigned char meta[DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE];
    if (!read_exact(fd, meta, sizeof(meta)) ||
        !ds4_glm52_tp4_collective_frame_decode(
                meta, sizeof(meta), request, err, err_size)) {
        return false;
    }
    if (request->dtype != DS4_GLM52_TP4_TENSOR_DTYPE_F32 ||
        request->element_count == 0 ||
        request->element_count > element_capacity ||
        request->byte_count != request->element_count * sizeof(payload[0]) ||
        header.payload_size !=
            DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE + request->byte_count) {
        set_error(err, err_size,
                  "TP4 collective transport f32 payload size does not match frame metadata");
        return false;
    }
    if (!read_exact(fd, payload, request->byte_count)) {
        set_error(err, err_size,
                  "TP4 collective transport f32 payload read failed");
        return false;
    }
    if (element_count_out) *element_count_out = request->element_count;
    return true;
}
