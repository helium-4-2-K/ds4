#ifndef DS4_GLM52_MOCK_H
#define DS4_GLM52_MOCK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_glm52_l0.h"

#define DS4_GLM52_MOCK_N_LAYER 79
#define DS4_GLM52_MOCK_N_EMBD 6144
#define DS4_GLM52_MOCK_N_VOCAB 154880
#define DS4_GLM52_MOCK_N_HEAD 64
#define DS4_GLM52_MOCK_N_HEAD_DIM 576
#define DS4_GLM52_MOCK_N_VALUE_MLA 256
#define DS4_GLM52_MOCK_N_KV_LORA 512
#define DS4_GLM52_MOCK_N_QK_ROPE 64
#define DS4_GLM52_MOCK_N_EXPERT 256
#define DS4_GLM52_MOCK_N_EXPERT_USED 8
#define DS4_GLM52_MOCK_N_NEXTN 1
#define DS4_GLM52_MOCK_MAX_TENSORS 192
#define DS4_GLM52_MOCK_HIDDEN_DTYPE_F32 1
#define DS4_GLM52_MOCK_DCP_TOPK 4
#define DS4_GLM52_MOCK_DCP_MAX_ROWS \
    (DS4_GLM52_L0_RANK_COUNT * DS4_GLM52_MOCK_DCP_TOPK)

typedef struct {
    char name[64];
    ds4_glm52_layout_role role;
    ds4_glm52_layout_scope scope;
    int owner_rank;
    int shard_axis;
    int64_t dim[4];
    int ndim;
    int range_start;
    int range_end;
    uint64_t tensor_id;
} ds4_glm52_mock_tensor_desc;

typedef struct {
    int rank;
    int tp_size;
    int dcp_size;
    int pp_size;
    int q_head_start;
    int q_head_end;
    int vocab_start;
    int vocab_end;
    int tensor_count;
    uint64_t model_hash;
    ds4_glm52_mock_tensor_desc tensors[DS4_GLM52_MOCK_MAX_TENSORS];
    bool initialized;
} ds4_glm52_mock_model;

typedef struct {
    uint64_t session_hash;
    uint64_t token_step_j;
    uint64_t kv_length;
    uint64_t hidden_checksum;
    uint64_t logits_checksum;
    int last_token;
    bool prefilled;
} ds4_glm52_mock_session;

typedef struct {
    uint64_t session_hash;
    uint64_t token_step_j;
    uint64_t kv_length;
    uint64_t hidden_checksum;
    uint64_t logits_checksum;
    int last_token;
    bool prefilled;
} ds4_glm52_mock_session_state;

typedef struct {
    int rank;
    uint64_t model_hash;
    uint64_t session_hash;
    ds4_glm52_tp4_command command;
    uint64_t seq;
    int q_head_start;
    int q_head_end;
    int vocab_start;
    int vocab_end;
    int candidate_token;
    float candidate_score;
    uint64_t token_step_j;
    uint64_t kv_length;
    uint64_t hidden_checksum;
    uint64_t logits_checksum;
    bool present;
} ds4_glm52_mock_rank_step;

typedef struct {
    ds4_glm52_tp4_command command;
    uint64_t seq;
    uint64_t model_hash;
    uint64_t session_hash;
    uint64_t token_step_j;
    uint64_t kv_length;
    uint32_t rank_mask;
    int coordinator_token;
    int coordinator_rank;
    float coordinator_score;
    bool complete;
    ds4_glm52_mock_rank_step ranks[DS4_GLM52_L0_RANK_COUNT];
} ds4_glm52_mock_tp4_step;

typedef enum {
    DS4_GLM52_MOCK_ALLREDUCE_ATTN = 1,
    DS4_GLM52_MOCK_ALLREDUCE_FFN = 2,
} ds4_glm52_mock_allreduce_kind;

typedef struct {
    int rank;
    ds4_glm52_mock_allreduce_kind kind;
    uint64_t seq;
    uint64_t model_hash;
    uint64_t session_hash;
    uint64_t token_step_j;
    int layer;
    int dtype;
    uint64_t shape_hash;
    size_t hidden_count;
    const float *partial;
    bool present;
} ds4_glm52_mock_hidden_partial;

typedef struct {
    ds4_glm52_mock_allreduce_kind kind;
    uint64_t seq;
    uint64_t model_hash;
    uint64_t session_hash;
    uint64_t token_step_j;
    int layer;
    int dtype;
    uint64_t shape_hash;
    size_t hidden_count;
    uint32_t rank_mask;
    bool complete;
    const float *partials[DS4_GLM52_L0_RANK_COUNT];
} ds4_glm52_mock_allreduce_step;

typedef struct {
    int rank;
    uint64_t start_row;
    uint64_t end_row;
    bool present;
} ds4_glm52_mock_dcp_owner_range;

typedef struct {
    int owner_rank;
    int layer;
    uint64_t token_step_j;
    uint64_t row_id;
    uint64_t checksum;
    bool present;
} ds4_glm52_mock_dcp_row;

typedef struct {
    int requester_rank;
    uint64_t row_id;
    bool present;
} ds4_glm52_mock_dcp_request;

typedef struct {
    int requester_rank;
    int row_count;
    uint64_t checksum;
    ds4_glm52_mock_dcp_row rows[DS4_GLM52_MOCK_DCP_MAX_ROWS];
    bool present;
} ds4_glm52_mock_dcp_selection;

bool ds4_glm52_mock_model_init(const ds4_glm52_l0_config *cfg,
                               ds4_glm52_mock_model *model,
                               char *err,
                               size_t err_size);
bool ds4_glm52_mock_model_validate(const ds4_glm52_mock_model *model,
                                   char *err,
                                   size_t err_size);
bool ds4_glm52_mock_prefill(const ds4_glm52_mock_model *model,
                            const int *tokens,
                            size_t token_count,
                            ds4_glm52_mock_session *session,
                            float *logits,
                            size_t logits_count,
                            char *err,
                            size_t err_size);
bool ds4_glm52_mock_decode(const ds4_glm52_mock_model *model,
                           ds4_glm52_mock_session *session,
                           int input_token,
                           float *logits,
                           size_t logits_count,
                           char *err,
                           size_t err_size);
bool ds4_glm52_mock_tp4_prefill(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const int *tokens,
        size_t token_count,
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size);
bool ds4_glm52_mock_tp4_decode(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        int input_token,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_tp4_apply_collectives(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size);
bool ds4_glm52_mock_tp4_collective_step(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_tp4_step *step,
        int last_token,
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size);
int ds4_glm52_mock_next_token(const ds4_glm52_mock_session *session);
bool ds4_glm52_mock_rank_vocab_candidate(
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        int *token_out,
        float *score_out,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_tp4_step_add_rank(
        ds4_glm52_mock_tp4_step *step,
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_make_rank_step(
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_tp4_step_add_contribution(
        ds4_glm52_mock_tp4_step *step,
        const ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size);
uint64_t ds4_glm52_mock_hidden_shape_hash(size_t hidden_count, int dtype);
bool ds4_glm52_mock_make_hidden_partial(
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_mock_allreduce_kind kind,
        uint64_t seq,
        int layer,
        const float *partial,
        size_t hidden_count,
        ds4_glm52_mock_hidden_partial *out,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_allreduce_add_contribution(
        ds4_glm52_mock_allreduce_step *step,
        const ds4_glm52_mock_hidden_partial *partial,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_allreduce_finish(
        ds4_glm52_mock_allreduce_step *step,
        float *out,
        size_t out_count,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_dcp_build_owner_ranges(
        uint64_t row_count,
        ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size);
uint64_t ds4_glm52_mock_dcp_row_checksum(int owner_rank,
                                         int layer,
                                         uint64_t token_step_j,
                                         uint64_t row_id);
bool ds4_glm52_mock_dcp_exchange(
        const ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_dcp_row *rows,
        size_t row_count,
        const ds4_glm52_mock_dcp_request *requests,
        size_t request_count,
        ds4_glm52_mock_dcp_selection selections[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size);
bool ds4_glm52_mock_transport_send_rank_step(
        int fd,
        const ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_transport_recv_rank_step(
        int fd,
        ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_transport_send_tokens(
        int fd,
        const int *tokens,
        size_t token_count,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_transport_recv_tokens(
        int fd,
        int *tokens,
        size_t max_tokens,
        size_t *token_count,
        char *err,
        size_t err_size);
void ds4_glm52_mock_session_export(
        const ds4_glm52_mock_session *session,
        ds4_glm52_mock_session_state *state);
void ds4_glm52_mock_session_import(
        ds4_glm52_mock_session *session,
        const ds4_glm52_mock_session_state *state);
bool ds4_glm52_mock_transport_send_session_state(
        int fd,
        const ds4_glm52_mock_session_state *state,
        char *err,
        size_t err_size);
bool ds4_glm52_mock_transport_recv_session_state(
        int fd,
        ds4_glm52_mock_session_state *state,
        char *err,
        size_t err_size);
void ds4_glm52_mock_tokenize(const char *text, ds4_tokens *out);
char *ds4_glm52_mock_token_text(int token, size_t *len);

#endif
