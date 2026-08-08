#ifndef DS4_GLM52_L0_H
#define DS4_GLM52_L0_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

#define DS4_GLM52_L0_GRAPH_ID "serve"
#define DS4_GLM52_MODEL_LOAD_GRAPH_ID "model-load"
#define DS4_GLM52_MODEL_SHARD_LAYOUT_GRAPH_ID "model-shard-layout"
#define DS4_GLM52_PREFILL_GRAPH_ID "prefill"
#define DS4_GLM52_L0_KV_CACHE_DTYPE "fp8_ds_mla"
#define DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS 4096
#define DS4_GLM52_L0_PREFILL_CHUNK 1024
#define DS4_GLM52_L0_TP_SIZE 4
#define DS4_GLM52_L0_DCP_SIZE 4
#define DS4_GLM52_L0_PP_SIZE 1
#define DS4_GLM52_L0_RANK_COUNT 4
#define DS4_GLM52_L0_Q_HEADS 64
#define DS4_GLM52_L0_Q_HEADS_PER_RANK \
    (DS4_GLM52_L0_Q_HEADS / DS4_GLM52_L0_TP_SIZE)
#define DS4_GLM52_L0_EXPERTS 256
#define DS4_GLM52_L0_EXPERTS_PER_RANK \
    (DS4_GLM52_L0_EXPERTS / DS4_GLM52_L0_TP_SIZE)
#define DS4_GLM52_L0_VOCAB_SIZE 154880
#define DS4_GLM52_L0_EXPECTED_BASE_SHARDS 20
#define DS4_GLM52_L0_EXPECTED_MTP_SHARDS 1
#define DS4_GLM52_LAYOUT_FORMAT_VERSION "ds4-shard-layout/v1"
#define DS4_GLM52_LAYOUT_MAX_ENTRIES 512
#define DS4_GLM52_LAYOUT_MAX_LINE 1024
#define DS4_GLM52_LAYOUT_FIELD_MAX 256
#define DS4_GLM52_L0_FABRIC_DATA_PLANE "crs812-200g"

typedef enum {
    DS4_GLM52_L0_STATUS_OK = 0,
    DS4_GLM52_L0_STATUS_NOT_READY = 1,
    DS4_GLM52_L0_STATUS_INVALID = 2,
} ds4_glm52_l0_status;

typedef enum {
    DS4_GLM52_L0_ACTION_SERVE_OPEN = 0,
    DS4_GLM52_L0_ACTION_TP_GROUP,
    DS4_GLM52_L0_ACTION_ACCEPT,
    DS4_GLM52_L0_ACTION_PREFILL,
    DS4_GLM52_L0_ACTION_DECODE_TOKEN,
    DS4_GLM52_L0_ACTION_STREAM_TOKEN,
    DS4_GLM52_L0_ACTION_KV_CHECKPOINT,
    DS4_GLM52_L0_ACTION_COUNT,
} ds4_glm52_l0_action;

typedef enum {
    DS4_GLM52_MODEL_LOAD_VALIDATE_LAUNCH_PLAN = 0,
    DS4_GLM52_MODEL_LOAD_VALIDATE_SHARD_MANIFEST,
    DS4_GLM52_MODEL_LOAD_MAP_RANK_SHARDS,
    DS4_GLM52_MODEL_LOAD_READY_RANK_ENGINES,
    DS4_GLM52_MODEL_LOAD_ACTION_COUNT,
} ds4_glm52_model_load_action;

typedef struct {
    bool enabled;
    bool mock_model;
    bool mock_matmul;
    bool legacy_cuda_tensor_parallel;
    bool legacy_two_rank_tp;
    bool rank_set;
    int rank;
    int tp_size;
    int dcp_size;
    int pp_size;
    const char *model_root;
    const char *rank_plan;
    const char *layout_path;
    const char *fabric_addr;
} ds4_glm52_l0_config;

typedef struct {
    int tp_size;
    int dcp_size;
    int pp_size;
    int rank_count;
    int rank;
    int host_count;
    char model_name[64];
    char checkpoint_root[256];
    char fabric_addr[128];
    char fabric_data_plane[64];
    bool validated;
} ds4_glm52_l0_launch_plan;

typedef struct {
    int rank;
    int base_shard_count;
    int mtp_shard_count;
    uint64_t total_bytes;
    bool has_foreign_rank_shard;
    bool validated;
} ds4_glm52_l0_shard_manifest;

typedef struct {
    const char *model_root;
    const char *rank_plan_path;
    int tp_size;
    int dcp_size;
    int pp_size;
    bool validated;
} ds4_glm52_l0_model_plan;

typedef struct {
    int rank;
    const char *checkpoint_root;
    int base_shard_count;
    int mtp_shard_count;
    uint64_t mapped_bytes;
    bool mapped;
    bool no_foreign_rank_shard;
} ds4_glm52_l0_resident_rank_shards;

typedef struct {
    int rank;
    int q_head_start;
    int q_head_end;
    int dcp_rank;
    int expert_start;
    int expert_end;
    int vocab_start;
    int vocab_end;
    bool bound;
} ds4_glm52_l0_rank_plan;

typedef struct {
    int tp_size;
    int dcp_size;
    int pp_size;
    const char *fabric_addr;
    const char *fabric_data_plane;
    int rank_count;
    int local_rank;
    bool topology_bound;
    bool transport_ready;
    bool group_ready;
} ds4_glm52_l0_tp_fabric;

typedef struct {
    const char *session_id;
    uint64_t length;
    bool tp_aware;
    bool append_ordered;
    bool prefill_complete;
} ds4_glm52_l0_kv_state;

typedef enum {
    DS4_GLM52_L0_CURSOR_PHASE_NONE = 0,
    DS4_GLM52_L0_CURSOR_PHASE_PREFILL,
    DS4_GLM52_L0_CURSOR_PHASE_DECODE,
} ds4_glm52_l0_cursor_phase;

typedef struct {
    uint64_t token_step_j;
    uint64_t kv_length;
    bool replicated;
    ds4_glm52_l0_cursor_phase phase;
} ds4_glm52_l0_kv_cursor;

/* BCD prefill: d-prompt-tokens (a-tokenize-prompt output), leader-owned prompt
 * token span for one accepted request. */
typedef struct {
    char session_id[128];
    int token_ids[DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS];
    int prompt_length;
    int consumed;
    bool set;
} ds4_glm52_l0_prompt_tokens;

/* BCD prefill: d-prefill-chunk (a-plan-prefill-chunks output), replicated via
 * the serve prefill action. start_position/end_position are exclusive prompt
 * positions ([start,end)); token ids stay in the prompt span. */
typedef struct {
    int chunk_index;
    int start_position;
    int end_position;
    int token_count;
} ds4_glm52_l0_prefill_chunk;

/* BCD prefill: d-prefill-kv-commit / d-prefill-handoff (a-commit-prefill-kv /
 * a-prefill-ready outputs), replicated decode-handoff summary. */
typedef struct {
    uint64_t prefilled_length;
    int tp_degree;
    int dcp_degree;
    const char *kv_cache_dtype;
    bool committed;
} ds4_glm52_l0_prefill_commit;

typedef struct {
    const char *session_id;
    bool accepted;
} ds4_glm52_l0_request_session;

typedef struct {
    int token_id;
    uint64_t position;
    bool present;
} ds4_glm52_l0_token;

typedef struct {
    int owner_rank;
    bool coordinator_visible;
} ds4_glm52_l0_logits_summary;

typedef struct {
    bool optional;
    bool requested;
    bool complete;
} ds4_glm52_l0_checkpoint_status;

typedef struct {
    ds4_glm52_l0_launch_plan launch_plan;
    ds4_glm52_l0_shard_manifest shard_manifest;
    ds4_glm52_l0_model_plan model_plan;
    ds4_glm52_l0_resident_rank_shards resident_shards;
    ds4_glm52_l0_rank_plan rank_plan;
    ds4_glm52_l0_tp_fabric tp_fabric;
    ds4_glm52_l0_kv_state kv;
    ds4_glm52_l0_kv_cursor cursor;
    ds4_glm52_l0_request_session request;
    ds4_glm52_l0_token token;
    ds4_glm52_l0_logits_summary logits;
    ds4_glm52_l0_checkpoint_status checkpoint;
    ds4_glm52_l0_prompt_tokens prompt;
    ds4_glm52_l0_prefill_chunk chunk;
    ds4_glm52_l0_prefill_commit commit;
} ds4_glm52_l0_state;

typedef struct {
    ds4_glm52_l0_status status;
    ds4_glm52_l0_action action;
    const char *graph_id;
    const char *action_id;
    char message[192];
} ds4_glm52_l0_result;

typedef void (*ds4_glm52_l0_trace_fn)(void *ud,
                                      const char *graph_id,
                                      const char *action_id,
                                      ds4_glm52_l0_status status,
                                      bool prerequisite_blocked,
                                      const ds4_glm52_l0_state *state);

size_t ds4_glm52_l0_action_count(void);
const char *ds4_glm52_l0_action_id(ds4_glm52_l0_action action);
const char *ds4_glm52_model_load_action_id(ds4_glm52_model_load_action action);
const char *ds4_glm52_l0_status_name(ds4_glm52_l0_status status);

bool ds4_glm52_l0_config_from_engine(const ds4_engine_options *opt,
                                     ds4_glm52_l0_config *cfg,
                                     char *err,
                                     size_t err_size);
bool ds4_glm52_l0_validate_config(const ds4_glm52_l0_config *cfg,
                                  char *err,
                                  size_t err_size);
ds4_glm52_l0_status ds4_glm52_l0_stub_action(ds4_glm52_l0_action action,
                                             const ds4_glm52_l0_config *cfg,
                                             ds4_glm52_l0_state *state,
                                             ds4_glm52_l0_result *result);
ds4_glm52_l0_status ds4_glm52_l0_run_skeleton(const ds4_glm52_l0_config *cfg,
                                             ds4_glm52_l0_state *state,
                                             ds4_glm52_l0_result *result,
                                             ds4_glm52_l0_trace_fn trace,
                                             void *trace_ud);
ds4_glm52_l0_status ds4_glm52_l0_review_skeleton(const ds4_glm52_l0_config *cfg,
                                                ds4_glm52_l0_state *state,
                                                ds4_glm52_l0_result *result,
                                                ds4_glm52_l0_trace_fn trace,
                                                void *trace_ud);

/* ---- BCD prefill child graph (serve/action-serve-prefill expands here) ---- */
/* prefill/a-tokenize-prompt -> a-plan-prefill-chunks -> a-prefill-layer-tp ->
 * a-prefill-allreduce -> a-commit-prefill-kv -> a-prefill-ready */

size_t ds4_glm52_prefill_action_count(void);
const char *ds4_glm52_prefill_action_id(int action_index);
const char *ds4_glm52_l0_cursor_phase_name(ds4_glm52_l0_cursor_phase phase);

/* a-tokenize-prompt seam: bind the leader-owned prompt token span for the
 * accepted session. Rejects a missing session, a missing span, or a span
 * beyond the L0 prompt bound. The span is copied into state; token_count may
 * be 0 for a deterministic empty-prompt prefill. */
bool ds4_glm52_l0_prefill_set_prompt(ds4_glm52_l0_state *state,
                                     const char *session_id,
                                     const int *token_ids,
                                     int token_count,
                                     char *err,
                                     size_t err_size);

/* Run the whole prefill child graph for the accepted request. Preconditions
 * (resident rank-local shards, ready TP4/DCP4 fabric, accepted request) block
 * with NOT_READY before any cursor mutation. Rank-plan/model shard identity,
 * KV session identity, TP-aware append-ordered KV identity, and append-order
 * are validated before any state is committed. chunk_size <= 0 uses the BCD
 * reference prefill chunk (1024). On success the prompt prefix is committed
 * to state-kv and state-kv-cursor is left decode-ready at the prompt length. */
ds4_glm52_l0_status ds4_glm52_l0_prefill_run(
        const ds4_glm52_l0_config *cfg,
        ds4_glm52_l0_state *state,
        int chunk_size,
        ds4_glm52_l0_trace_fn trace,
        void *trace_ud,
        ds4_glm52_l0_result *result);

/* ---- Production execution frontiers for GLM5.2 TP4/DCP4 ----
 *
 * These are strict real-path contracts, not mock math. They validate the
 * rank/session/model/cursor/shape evidence that a production backend must
 * supply, then fail closed with NOT_READY until the concrete GPU/network
 * implementation is wired behind the same boundary. */

typedef enum {
    DS4_GLM52_TP4_COLLECTIVE_ATTN = 1,
    DS4_GLM52_TP4_COLLECTIVE_FFN = 2,
    DS4_GLM52_TP4_COLLECTIVE_LOGITS = 3,
} ds4_glm52_tp4_collective_kind;

typedef struct {
    ds4_glm52_tp4_collective_kind kind;
    int rank;
    int tp_size;
    int dcp_size;
    int rank_count;
    int layer_index;
    int dtype;
    uint64_t seq;
    uint64_t model_hash;
    uint64_t session_hash;
    uint64_t token_step_j;
    size_t element_count;
    uint32_t participant_mask;
    bool topology_ready;
    bool transport_ready;
    bool rank_local_partial_ready;
    bool replicated_output_ready;
} ds4_glm52_tp4_collective_request;

typedef struct {
    int rank;
    int dcp_size;
    int rank_count;
    int layer_index;
    uint64_t seq;
    uint64_t model_hash;
    uint64_t session_hash;
    uint64_t token_step_j;
    int selected_row_count;
    uint32_t owner_rank_mask;
    bool ownership_plan_valid;
    bool append_ordered_kv;
    bool row_payload_ready;
    bool transport_ready;
} ds4_glm52_dcp_exchange_request;

typedef struct {
    int rank;
    int input_token;
    uint64_t seq;
    uint64_t model_hash;
    uint64_t session_hash;
    bool model_ready;
    bool tp_collectives_ready;
    bool dcp_exchange_ready;
    bool glm52_kernels_ready;
} ds4_glm52_decode_real_request;

const char *ds4_glm52_tp4_collective_kind_name(
        ds4_glm52_tp4_collective_kind kind);
ds4_glm52_l0_status ds4_glm52_tp4_real_collective_allreduce(
        const ds4_glm52_tp4_collective_request *request,
        ds4_glm52_l0_result *result);
ds4_glm52_l0_status ds4_glm52_dcp_real_row_exchange(
        const ds4_glm52_dcp_exchange_request *request,
        ds4_glm52_l0_result *result);
ds4_glm52_l0_status ds4_glm52_decode_real_step(
        const ds4_glm52_l0_config *cfg,
        const ds4_glm52_decode_real_request *request,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result);

/* ---- TP4 no-model rank-group handshake seam ---- */

typedef enum {
    DS4_GLM52_TP4_COMMAND_NONE = 0,
    DS4_GLM52_TP4_COMMAND_LOAD,
    DS4_GLM52_TP4_COMMAND_PREFILL,
    DS4_GLM52_TP4_COMMAND_DECODE,
    DS4_GLM52_TP4_COMMAND_SHUTDOWN,
} ds4_glm52_tp4_command;

typedef struct {
    int rank;
    int tp_size;
    int dcp_size;
    uint64_t model_hash;
    uint64_t config_hash;
    uint64_t plan_hash;
    bool registered;
    bool command_ack;
} ds4_glm52_tp4_rank_member;

typedef struct {
    int rank_count;
    int registered_count;
    uint32_t registered_mask;
    uint32_t ack_mask;
    uint64_t model_hash;
    uint64_t config_hash;
    uint64_t plan_hash;
    uint64_t command_seq;
    ds4_glm52_tp4_command command;
    ds4_glm52_tp4_rank_member members[DS4_GLM52_L0_RANK_COUNT];
    bool topology_ready;
    bool transport_ready;
    bool group_ready;
} ds4_glm52_tp4_rank_group;

void ds4_glm52_tp4_rank_group_init(ds4_glm52_tp4_rank_group *group,
                                   uint64_t model_hash,
                                   uint64_t config_hash,
                                   uint64_t plan_hash);
bool ds4_glm52_tp4_rank_group_register(
        ds4_glm52_tp4_rank_group *group,
        int rank,
        int tp_size,
        int dcp_size,
        uint64_t model_hash,
        uint64_t config_hash,
        uint64_t plan_hash,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_rank_group_ready(const ds4_glm52_tp4_rank_group *group);
bool ds4_glm52_tp4_rank_group_broadcast(
        ds4_glm52_tp4_rank_group *group,
        int coordinator_rank,
        ds4_glm52_tp4_command command,
        uint64_t *seq_out,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_rank_group_ack(
        ds4_glm52_tp4_rank_group *group,
        int rank,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_rank_group_command_done(
        const ds4_glm52_tp4_rank_group *group);
bool ds4_glm52_tp4_rank_group_publish_fabric(
        const ds4_glm52_tp4_rank_group *group,
        int local_rank,
        ds4_glm52_l0_state *state,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_rank_group_transport_failed(
        ds4_glm52_tp4_rank_group *group,
        int rank,
        char *err,
        size_t err_size);
const char *ds4_glm52_tp4_command_name(ds4_glm52_tp4_command command);

typedef struct {
    int rank;
    int tp_size;
    int dcp_size;
    uint64_t model_hash;
    uint64_t config_hash;
    uint64_t plan_hash;
} ds4_glm52_tp4_transport_hello;

typedef struct {
    char host[128];
    uint16_t port;
} ds4_glm52_tp4_tcp_endpoint;

bool ds4_glm52_tp4_transport_send_hello(
        int fd,
        const ds4_glm52_tp4_transport_hello *hello,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_transport_recv_hello_and_register(
        int fd,
        ds4_glm52_tp4_rank_group *group,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_transport_send_command(
        int fd,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_transport_recv_command(
        int fd,
        ds4_glm52_tp4_command *command,
        uint64_t *seq,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_transport_send_ack(
        int fd,
        int rank,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_transport_recv_ack_and_record(
        int fd,
        ds4_glm52_tp4_rank_group *group,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_tcp_parse_endpoint(
        const char *addr,
        ds4_glm52_tp4_tcp_endpoint *endpoint,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_tcp_listen(
        const ds4_glm52_tp4_tcp_endpoint *endpoint,
        int *listen_fd,
        uint16_t *bound_port,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_tcp_accept(
        int listen_fd,
        int timeout_ms,
        int *accepted_fd,
        char *err,
        size_t err_size);
bool ds4_glm52_tp4_tcp_connect(
        const ds4_glm52_tp4_tcp_endpoint *endpoint,
        int timeout_ms,
        int *connected_fd,
        char *err,
        size_t err_size);

/* ---- model-shard-layout child: ds4_shard_layout types ---- */

typedef enum {
    DS4_GLM52_LAYOUT_ROLE_REPLICATED = 0,
    DS4_GLM52_LAYOUT_ROLE_Q_HEAD,
    DS4_GLM52_LAYOUT_ROLE_MLA_KV,
    DS4_GLM52_LAYOUT_ROLE_EXPERT,
    DS4_GLM52_LAYOUT_ROLE_VOCAB,
    DS4_GLM52_LAYOUT_ROLE_OUTPUT,
    DS4_GLM52_LAYOUT_ROLE_UNKNOWN,
} ds4_glm52_layout_role;

typedef enum {
    DS4_GLM52_LAYOUT_SCOPE_REPLICATED = 0,
    DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL,
} ds4_glm52_layout_scope;

typedef struct {
    char tensor_name[DS4_GLM52_LAYOUT_FIELD_MAX];
    ds4_glm52_layout_role role;
    ds4_glm52_layout_scope scope;
    int rank;
    char file_path[DS4_GLM52_LAYOUT_FIELD_MAX];
    uint64_t byte_offset;
    uint64_t byte_length;
    char sha256[DS4_GLM52_LAYOUT_FIELD_MAX];
    bool replicated;
    int q_head_start;
    int q_head_end;
    int expert_start;
    int expert_end;
    int vocab_start;
    int vocab_end;
} ds4_glm52_layout_entry;

typedef struct {
    char format_version[DS4_GLM52_LAYOUT_FIELD_MAX];
    char model_config_sha256[DS4_GLM52_LAYOUT_FIELD_MAX];
    int tp_size;
    int dcp_size;
    int rank_count;
    int required_base_shards;
    int required_mtp_shards;
    bool has_mtp;
    int entry_count;
    ds4_glm52_layout_entry entries[DS4_GLM52_LAYOUT_MAX_ENTRIES];
    bool validated;
} ds4_glm52_layout_spec;

typedef struct {
    int rank;
    int q_head_start;
    int q_head_end;
    int expert_start;
    int expert_end;
    int vocab_start;
    int vocab_end;
    bool coverage_complete;
    bool overlaps_present;
    bool missing_spans_present;
    int entry_count;
    ds4_glm52_layout_entry entries[DS4_GLM52_LAYOUT_MAX_ENTRIES];
    bool validated;
} ds4_glm52_layout_ownership_plan;

typedef struct {
    int rank;
    int tensor_count;
    uint64_t mapped_bytes;
    char source_layout_sha256[DS4_GLM52_LAYOUT_FIELD_MAX];
    bool no_foreign_rank_shard;
    bool hash_verified;
    bool mapped;
} ds4_glm52_layout_mapped_slices;

/* BCD code unit: a-read-layout-spec */
ds4_glm52_l0_status ds4_glm52_layout_read_manifest(
        const char *layout_path,
        ds4_glm52_layout_spec *spec,
        ds4_glm52_l0_result *result);

/* BCD code unit: a-validate-tensor-ownership */
ds4_glm52_l0_status ds4_glm52_layout_validate_ownership(
        const ds4_glm52_layout_spec *spec,
        int current_rank,
        ds4_glm52_layout_ownership_plan *plan,
        ds4_glm52_l0_result *result);

/* BCD code unit: a-load-rank-tensors */
ds4_glm52_l0_status ds4_glm52_layout_mmap_rank_tensors(
        const ds4_glm52_layout_ownership_plan *plan,
        const char *checkpoint_root,
        ds4_glm52_layout_mapped_slices *mapped,
        ds4_glm52_l0_result *result);

/* BCD code unit: a-publish-loaded-shard */
ds4_glm52_l0_status ds4_glm52_layout_publish_loaded_rank_shard(
        const ds4_glm52_layout_mapped_slices *mapped,
        const ds4_glm52_layout_ownership_plan *plan,
        ds4_glm52_l0_state *state,
        ds4_glm52_l0_result *result);

const char *ds4_glm52_layout_role_name(ds4_glm52_layout_role role);

#endif
