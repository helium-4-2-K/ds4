#include "ds4_glm52_mock.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DS4_GLM52_MOCK_STEP_MAGIC UINT32_C(0x44534d35)
#define DS4_GLM52_MOCK_STEP_VERSION UINT32_C(1)

static void mock_error(char *err, size_t err_size, const char *msg) {
    if (!err || err_size == 0) return;
    snprintf(err, err_size, "%s", msg ? msg : "unknown GLM 5.2 mock error");
}

static uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        h ^= (uint8_t)(v & 0xffu);
        h *= 1099511628211ull;
        v >>= 8;
    }
    return h;
}

static uint64_t fnv1a_bytes(uint64_t h, const char *s) {
    if (!s) return h;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= *p;
        h *= 1099511628211ull;
    }
    return h;
}

static bool mock_float_ptr_isfinite(const float *v) {
    uint32_t bits;
    memcpy(&bits, v, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static bool add_tensor(ds4_glm52_mock_model *model,
                       const char *name,
                       ds4_glm52_layout_role role,
                       ds4_glm52_layout_scope scope,
                       int owner_rank,
                       int shard_axis,
                       int64_t d0,
                       int64_t d1,
                       int64_t d2,
                       int64_t d3,
                       int range_start,
                       int range_end) {
    if (model->tensor_count >= DS4_GLM52_MOCK_MAX_TENSORS) return false;
    ds4_glm52_mock_tensor_desc *t = &model->tensors[model->tensor_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->role = role;
    t->scope = scope;
    t->owner_rank = owner_rank;
    t->shard_axis = shard_axis;
    t->dim[0] = d0;
    t->dim[1] = d1;
    t->dim[2] = d2;
    t->dim[3] = d3;
    t->ndim = d3 ? 4 : d2 ? 3 : d1 ? 2 : 1;
    t->range_start = range_start;
    t->range_end = range_end;
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_bytes(h, name);
    h = fnv1a_u64(h, (uint64_t)role);
    h = fnv1a_u64(h, (uint64_t)(owner_rank + 1));
    h = fnv1a_u64(h, (uint64_t)(range_start + 1));
    h = fnv1a_u64(h, (uint64_t)(range_end + 1));
    t->tensor_id = h;
    return true;
}

bool ds4_glm52_mock_model_init(const ds4_glm52_l0_config *cfg,
                               ds4_glm52_mock_model *model,
                               char *err,
                               size_t err_size) {
    if (!cfg || !model) {
        mock_error(err, err_size, "GLM 5.2 mock init requires config and output model");
        return false;
    }
    if (!cfg->mock_model || !cfg->mock_matmul) {
        mock_error(err, err_size, "GLM 5.2 mock model requires explicit mock_model and mock_matmul");
        return false;
    }
    if (!ds4_glm52_l0_validate_config(cfg, err, err_size)) return false;

    memset(model, 0, sizeof(*model));
    model->rank = cfg->rank;
    model->tp_size = cfg->tp_size;
    model->dcp_size = cfg->dcp_size;
    model->pp_size = cfg->pp_size;
    model->q_head_start = cfg->rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
    model->q_head_end = model->q_head_start + DS4_GLM52_L0_Q_HEADS_PER_RANK;
    model->vocab_start =
        (DS4_GLM52_MOCK_N_VOCAB * cfg->rank) / DS4_GLM52_L0_TP_SIZE;
    model->vocab_end =
        (DS4_GLM52_MOCK_N_VOCAB * (cfg->rank + 1)) / DS4_GLM52_L0_TP_SIZE;
    model->model_hash = 1469598103934665603ull;
    model->model_hash = fnv1a_bytes(model->model_hash, "glm52-mock/full-shape/tp4");

    const int q_owned = model->q_head_end - model->q_head_start;
    if (!add_tensor(model, "token_embd.weight",
                    DS4_GLM52_LAYOUT_ROLE_REPLICATED,
                    DS4_GLM52_LAYOUT_SCOPE_REPLICATED, -1, -1,
                    DS4_GLM52_MOCK_N_EMBD, DS4_GLM52_MOCK_N_VOCAB, 0, 0,
                    0, DS4_GLM52_MOCK_N_VOCAB) ||
        !add_tensor(model, "output_norm.weight",
                    DS4_GLM52_LAYOUT_ROLE_REPLICATED,
                    DS4_GLM52_LAYOUT_SCOPE_REPLICATED, -1, -1,
                    DS4_GLM52_MOCK_N_EMBD, 0, 0, 0,
                    0, DS4_GLM52_MOCK_N_EMBD) ||
        !add_tensor(model, "output.weight.vocab_shard",
                    DS4_GLM52_LAYOUT_ROLE_VOCAB,
                    DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL, cfg->rank, 1,
                    DS4_GLM52_MOCK_N_EMBD,
                    model->vocab_end - model->vocab_start, 0, 0,
                    model->vocab_start, model->vocab_end)) {
        mock_error(err, err_size, "GLM 5.2 mock tensor descriptor capacity exceeded");
        return false;
    }

    for (int layer = 0; layer < DS4_GLM52_MOCK_N_LAYER; layer++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.attn_q_heads", layer);
        if (!add_tensor(model, name,
                        DS4_GLM52_LAYOUT_ROLE_Q_HEAD,
                        DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL, cfg->rank, 2,
                        DS4_GLM52_MOCK_N_EMBD,
                        DS4_GLM52_MOCK_N_HEAD_DIM,
                        q_owned, 0,
                        model->q_head_start, model->q_head_end)) {
            mock_error(err, err_size, "GLM 5.2 mock tensor descriptor capacity exceeded");
            return false;
        }
        snprintf(name, sizeof(name), "blk.%d.attn_mla_kv", layer);
        if (!add_tensor(model, name,
                        DS4_GLM52_LAYOUT_ROLE_MLA_KV,
                        DS4_GLM52_LAYOUT_SCOPE_REPLICATED, -1, -1,
                        DS4_GLM52_MOCK_N_EMBD,
                        DS4_GLM52_MOCK_N_KV_LORA + DS4_GLM52_MOCK_N_QK_ROPE,
                        0, 0,
                        0,
                        DS4_GLM52_MOCK_N_KV_LORA + DS4_GLM52_MOCK_N_QK_ROPE)) {
            mock_error(err, err_size, "GLM 5.2 mock tensor descriptor capacity exceeded");
            return false;
        }
    }
    model->initialized = true;
    return ds4_glm52_mock_model_validate(model, err, err_size);
}

bool ds4_glm52_mock_model_validate(const ds4_glm52_mock_model *model,
                                   char *err,
                                   size_t err_size) {
    if (!model || !model->initialized) {
        mock_error(err, err_size, "GLM 5.2 mock model is not initialized");
        return false;
    }
    if (model->tp_size != DS4_GLM52_L0_TP_SIZE ||
        model->dcp_size != DS4_GLM52_L0_DCP_SIZE ||
        model->pp_size != DS4_GLM52_L0_PP_SIZE ||
        model->rank < 0 ||
        model->rank >= DS4_GLM52_L0_RANK_COUNT) {
        mock_error(err, err_size, "GLM 5.2 mock model rank topology is invalid");
        return false;
    }
    if (model->q_head_end - model->q_head_start !=
        DS4_GLM52_L0_Q_HEADS_PER_RANK) {
        mock_error(err, err_size, "GLM 5.2 mock model Q-head shard is invalid");
        return false;
    }
    if (model->vocab_start < 0 ||
        model->vocab_end <= model->vocab_start ||
        model->vocab_end > DS4_GLM52_MOCK_N_VOCAB) {
        mock_error(err, err_size, "GLM 5.2 mock model vocab shard is invalid");
        return false;
    }
    return true;
}

static void mock_fill_logits(const ds4_glm52_mock_model *model,
                             ds4_glm52_mock_session *session,
                             float *logits,
                             size_t logits_count) {
    uint64_t h = session->hidden_checksum;
    h = fnv1a_u64(h, model ? (uint64_t)model->rank : 0);
    h = fnv1a_u64(h, session->token_step_j);
    session->logits_checksum = h;
    if (!logits || logits_count == 0) return;
    for (size_t i = 0; i < logits_count; i++) logits[i] = -120.0f;
    const size_t vocab = logits_count < DS4_GLM52_MOCK_N_VOCAB ?
        logits_count : DS4_GLM52_MOCK_N_VOCAB;
    const int best = (int)(h % (uint64_t)vocab);
    logits[best] = 12.0f;
    for (int i = 1; i <= 4; i++) {
        logits[(best + i) % (int)vocab] = 7.0f - (float)i;
    }
}

bool ds4_glm52_mock_prefill(const ds4_glm52_mock_model *model,
                            const int *tokens,
                            size_t token_count,
                            ds4_glm52_mock_session *session,
                            float *logits,
                            size_t logits_count,
                            char *err,
                            size_t err_size) {
    if (!tokens || token_count == 0 || !session) {
        mock_error(err, err_size, "GLM 5.2 mock prefill requires at least one token");
        return false;
    }
    if (!ds4_glm52_mock_model_validate(model, err, err_size)) return false;
    uint64_t h = fnv1a_u64(model->model_hash, token_count);
    for (size_t i = 0; i < token_count; i++) h = fnv1a_u64(h, (uint64_t)tokens[i]);
    memset(session, 0, sizeof(*session));
    session->session_hash = h;
    session->token_step_j = token_count;
    session->kv_length = token_count;
    session->hidden_checksum = h;
    session->last_token = tokens[token_count - 1u];
    session->prefilled = true;
    mock_fill_logits(model, session, logits, logits_count);
    return true;
}

bool ds4_glm52_mock_decode(const ds4_glm52_mock_model *model,
                           ds4_glm52_mock_session *session,
                           int input_token,
                           float *logits,
                           size_t logits_count,
                           char *err,
                           size_t err_size) {
    if (!session || !session->prefilled) {
        mock_error(err, err_size, "GLM 5.2 mock decode requires a prefilled session");
        return false;
    }
    if (!ds4_glm52_mock_model_validate(model, err, err_size)) return false;
    session->hidden_checksum = fnv1a_u64(session->hidden_checksum,
                                        (uint64_t)(uint32_t)input_token);
    session->hidden_checksum = fnv1a_u64(session->hidden_checksum,
                                        session->token_step_j);
    session->last_token = input_token;
    session->token_step_j++;
    session->kv_length++;
    mock_fill_logits(model, session, logits, logits_count);
    return true;
}

int ds4_glm52_mock_next_token(const ds4_glm52_mock_session *session) {
    if (!session || !session->prefilled) return 0;
    return (int)(session->logits_checksum % DS4_GLM52_MOCK_N_VOCAB);
}

bool ds4_glm52_mock_rank_vocab_candidate(
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        int *token_out,
        float *score_out,
        char *err,
        size_t err_size) {
    if (!session || !session->prefilled) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 candidate requires a prefilled session");
        return false;
    }
    if (!ds4_glm52_mock_model_validate(model, err, err_size)) return false;
    const int span = model->vocab_end - model->vocab_start;
    if (span <= 0) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 candidate has empty vocab shard");
        return false;
    }
    const int token = model->vocab_start +
        (int)(session->logits_checksum % (uint64_t)span);
    const float score =
        1.0f + (float)(session->logits_checksum % 100000u) * 0.00001f +
        (float)model->rank * 0.000001f;
    if (token_out) *token_out = token;
    if (score_out) *score_out = score;
    return true;
}

static bool mock_tp4_check_complete(ds4_glm52_mock_tp4_step *step,
                                    char *err,
                                    size_t err_size) {
    if (step->rank_mask != ((1u << DS4_GLM52_L0_RANK_COUNT) - 1u)) {
        step->complete = false;
        return true;
    }

    bool q_seen[DS4_GLM52_MOCK_N_HEAD] = {0};
    int vocab_end = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_mock_rank_step *r = &step->ranks[rank];
        if (!r->present || r->rank != rank) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 step missing rank state");
            return false;
        }
        const int expected_q0 = rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
        const int expected_q1 = expected_q0 + DS4_GLM52_L0_Q_HEADS_PER_RANK;
        if (r->q_head_start != expected_q0 || r->q_head_end != expected_q1) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 Q-head ownership gap");
            return false;
        }
        for (int q = r->q_head_start; q < r->q_head_end; q++) {
            if (q < 0 || q >= DS4_GLM52_MOCK_N_HEAD || q_seen[q]) {
                mock_error(err, err_size, "GLM 5.2 mock TP4 Q-head ownership overlap");
                return false;
            }
            q_seen[q] = true;
        }
        if (r->vocab_start != vocab_end || r->vocab_end <= r->vocab_start) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 vocab shard coverage gap");
            return false;
        }
        vocab_end = r->vocab_end;
    }
    for (int q = 0; q < DS4_GLM52_MOCK_N_HEAD; q++) {
        if (!q_seen[q]) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 Q-head coverage incomplete");
            return false;
        }
    }
    if (vocab_end != DS4_GLM52_MOCK_N_VOCAB) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 vocab coverage incomplete");
        return false;
    }
    step->complete = true;
    return true;
}

bool ds4_glm52_mock_make_rank_step(
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size) {
    if (!rank_step || !session || !session->prefilled) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 rank step requires rank session state");
        return false;
    }
    if (command != DS4_GLM52_TP4_COMMAND_PREFILL &&
        command != DS4_GLM52_TP4_COMMAND_DECODE) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 rank step only accepts prefill/decode");
        return false;
    }
    if (seq == 0) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 rank step requires nonzero command sequence");
        return false;
    }
    if (!ds4_glm52_mock_model_validate(model, err, err_size)) return false;

    int candidate = -1;
    float score = 0.0f;
    if (!ds4_glm52_mock_rank_vocab_candidate(model, session, &candidate,
                                             &score, err, err_size)) {
        return false;
    }

    memset(rank_step, 0, sizeof(*rank_step));
    rank_step->rank = model->rank;
    rank_step->model_hash = model->model_hash;
    rank_step->session_hash = session->session_hash;
    rank_step->command = command;
    rank_step->seq = seq;
    rank_step->q_head_start = model->q_head_start;
    rank_step->q_head_end = model->q_head_end;
    rank_step->vocab_start = model->vocab_start;
    rank_step->vocab_end = model->vocab_end;
    rank_step->candidate_token = candidate;
    rank_step->candidate_score = score;
    rank_step->token_step_j = session->token_step_j;
    rank_step->kv_length = session->kv_length;
    rank_step->hidden_checksum = session->hidden_checksum;
    rank_step->logits_checksum = session->logits_checksum;
    rank_step->present = true;
    return true;
}

bool ds4_glm52_mock_tp4_step_add_contribution(
        ds4_glm52_mock_tp4_step *step,
        const ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size) {
    if (!step || !rank_step || !rank_step->present) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 step requires a rank contribution");
        return false;
    }
    if (rank_step->command != DS4_GLM52_TP4_COMMAND_PREFILL &&
        rank_step->command != DS4_GLM52_TP4_COMMAND_DECODE) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 contribution only accepts prefill/decode");
        return false;
    }
    if (rank_step->seq == 0) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 contribution requires nonzero command sequence");
        return false;
    }
    const int rank = rank_step->rank;
    if (rank < 0 || rank >= DS4_GLM52_L0_RANK_COUNT) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 contribution rank is invalid");
        return false;
    }
    const int expected_q0 = rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
    const int expected_q1 = expected_q0 + DS4_GLM52_L0_Q_HEADS_PER_RANK;
    if (rank_step->q_head_start != expected_q0 ||
        rank_step->q_head_end != expected_q1) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 contribution Q-head ownership is invalid");
        return false;
    }
    const int expected_vocab0 =
        (DS4_GLM52_MOCK_N_VOCAB * rank) / DS4_GLM52_L0_TP_SIZE;
    const int expected_vocab1 =
        (DS4_GLM52_MOCK_N_VOCAB * (rank + 1)) / DS4_GLM52_L0_TP_SIZE;
    if (rank_step->vocab_start != expected_vocab0 ||
        rank_step->vocab_end != expected_vocab1 ||
        rank_step->candidate_token < rank_step->vocab_start ||
        rank_step->candidate_token >= rank_step->vocab_end) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 contribution vocab ownership is invalid");
        return false;
    }
    if (step->rank_mask == 0) {
        memset(step, 0, sizeof(*step));
        step->command = rank_step->command;
        step->seq = rank_step->seq;
        step->model_hash = rank_step->model_hash;
        step->session_hash = rank_step->session_hash;
        step->token_step_j = rank_step->token_step_j;
        step->kv_length = rank_step->kv_length;
        step->coordinator_token = -1;
        step->coordinator_rank = -1;
        step->coordinator_score = -1.0e30f;
    } else {
        if (step->command != rank_step->command) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 command mismatch");
            return false;
        }
        if (step->seq != rank_step->seq) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 sequence mismatch");
            return false;
        }
        if (step->model_hash != rank_step->model_hash) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 model identity mismatch");
            return false;
        }
        if (step->session_hash != rank_step->session_hash) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 session identity mismatch");
            return false;
        }
        if (step->token_step_j != rank_step->token_step_j ||
            step->kv_length != rank_step->kv_length) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 cursor mismatch");
            return false;
        }
    }
    if (step->rank_mask & (1u << rank)) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 duplicate rank contribution");
        return false;
    }

    ds4_glm52_mock_rank_step *r = &step->ranks[rank];
    *r = *rank_step;
    step->rank_mask |= 1u << rank;
    if (rank_step->candidate_score > step->coordinator_score ||
        (rank_step->candidate_score == step->coordinator_score &&
         (step->coordinator_token < 0 ||
          rank_step->candidate_token < step->coordinator_token))) {
        step->coordinator_score = rank_step->candidate_score;
        step->coordinator_token = rank_step->candidate_token;
        step->coordinator_rank = rank;
    }
    return mock_tp4_check_complete(step, err, err_size);
}

bool ds4_glm52_mock_tp4_step_add_rank(
        ds4_glm52_mock_tp4_step *step,
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size) {
    ds4_glm52_mock_rank_step rank_step;
    if (!ds4_glm52_mock_make_rank_step(model, session, command, seq,
                                       &rank_step, err, err_size)) {
        return false;
    }
    return ds4_glm52_mock_tp4_step_add_contribution(step, &rank_step,
                                                    err, err_size);
}

static bool mock_valid_allreduce_kind(ds4_glm52_mock_allreduce_kind kind) {
    return kind == DS4_GLM52_MOCK_ALLREDUCE_ATTN ||
           kind == DS4_GLM52_MOCK_ALLREDUCE_FFN;
}

uint64_t ds4_glm52_mock_hidden_shape_hash(size_t hidden_count, int dtype) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_bytes(h, "glm52-mock/hidden-shape");
    h = fnv1a_u64(h, (uint64_t)hidden_count);
    h = fnv1a_u64(h, (uint64_t)(uint32_t)dtype);
    return h;
}

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
        size_t err_size) {
    if (!out || !session || !session->prefilled) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce requires rank session state");
        return false;
    }
    if (!mock_valid_allreduce_kind(kind)) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce kind is invalid");
        return false;
    }
    if (seq == 0) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce requires nonzero command sequence");
        return false;
    }
    if (layer < 0 || layer >= DS4_GLM52_MOCK_N_LAYER) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce layer is invalid");
        return false;
    }
    if (!partial || hidden_count == 0) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce hidden partial is empty");
        return false;
    }
    if (hidden_count != DS4_GLM52_MOCK_N_EMBD) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce hidden shape is invalid");
        return false;
    }
    if (!ds4_glm52_mock_model_validate(model, err, err_size)) return false;

    for (size_t i = 0; i < hidden_count; i++) {
        if (!mock_float_ptr_isfinite(&partial[i])) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce partial contains non-finite values");
            return false;
        }
    }

    memset(out, 0, sizeof(*out));
    out->rank = model->rank;
    out->kind = kind;
    out->seq = seq;
    out->model_hash = model->model_hash;
    out->session_hash = session->session_hash;
    out->token_step_j = session->token_step_j;
    out->layer = layer;
    out->dtype = DS4_GLM52_MOCK_HIDDEN_DTYPE_F32;
    out->shape_hash =
        ds4_glm52_mock_hidden_shape_hash(hidden_count, out->dtype);
    out->hidden_count = hidden_count;
    out->partial = partial;
    out->present = true;
    return true;
}

static bool mock_allreduce_check_complete(ds4_glm52_mock_allreduce_step *step) {
    step->complete =
        step->rank_mask == ((1u << DS4_GLM52_L0_RANK_COUNT) - 1u);
    return step->complete;
}

bool ds4_glm52_mock_allreduce_add_contribution(
        ds4_glm52_mock_allreduce_step *step,
        const ds4_glm52_mock_hidden_partial *partial,
        char *err,
        size_t err_size) {
    if (!step || !partial || !partial->present || !partial->partial) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce requires a hidden partial contribution");
        return false;
    }
    if (!mock_valid_allreduce_kind(partial->kind)) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce contribution kind is invalid");
        return false;
    }
    if (partial->seq == 0) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce contribution requires nonzero sequence");
        return false;
    }
    if (partial->rank < 0 || partial->rank >= DS4_GLM52_L0_RANK_COUNT) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce contribution rank is invalid");
        return false;
    }
    if (partial->layer < 0 || partial->layer >= DS4_GLM52_MOCK_N_LAYER) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce contribution layer is invalid");
        return false;
    }
    if (partial->dtype != DS4_GLM52_MOCK_HIDDEN_DTYPE_F32 ||
        partial->hidden_count != DS4_GLM52_MOCK_N_EMBD ||
        partial->shape_hash != ds4_glm52_mock_hidden_shape_hash(
            partial->hidden_count, partial->dtype)) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce contribution shape/dtype is invalid");
        return false;
    }
    for (size_t i = 0; i < partial->hidden_count; i++) {
        if (!mock_float_ptr_isfinite(&partial->partial[i])) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce contribution contains non-finite values");
            return false;
        }
    }

    const int rank = partial->rank;
    if (step->rank_mask & (1u << rank)) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce duplicate rank contribution");
        return false;
    }
    if (step->rank_mask == 0) {
        memset(step, 0, sizeof(*step));
        step->kind = partial->kind;
        step->seq = partial->seq;
        step->model_hash = partial->model_hash;
        step->session_hash = partial->session_hash;
        step->token_step_j = partial->token_step_j;
        step->layer = partial->layer;
        step->dtype = partial->dtype;
        step->shape_hash = partial->shape_hash;
        step->hidden_count = partial->hidden_count;
    } else {
        if (step->kind != partial->kind) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce kind mismatch");
            return false;
        }
        if (step->seq != partial->seq) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce sequence mismatch");
            return false;
        }
        if (step->model_hash != partial->model_hash) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce model identity mismatch");
            return false;
        }
        if (step->session_hash != partial->session_hash) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce session identity mismatch");
            return false;
        }
        if (step->token_step_j != partial->token_step_j) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce token position mismatch");
            return false;
        }
        if (step->layer != partial->layer) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce layer mismatch");
            return false;
        }
        if (step->dtype != partial->dtype ||
            step->shape_hash != partial->shape_hash ||
            step->hidden_count != partial->hidden_count) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce shape/dtype mismatch");
            return false;
        }
    }

    step->partials[rank] = partial->partial;
    step->rank_mask |= 1u << rank;
    mock_allreduce_check_complete(step);
    return true;
}

bool ds4_glm52_mock_allreduce_finish(
        ds4_glm52_mock_allreduce_step *step,
        float *out,
        size_t out_count,
        char *err,
        size_t err_size) {
    if (!step || !out) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce finish requires output storage");
        return false;
    }
    if (!mock_allreduce_check_complete(step)) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce is missing rank contributions");
        return false;
    }
    if (out_count != step->hidden_count ||
        out_count != DS4_GLM52_MOCK_N_EMBD) {
        mock_error(err, err_size, "GLM 5.2 mock all-reduce output shape is invalid");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!step->partials[rank]) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce rank partial is missing");
            return false;
        }
    }
    for (size_t i = 0; i < out_count; i++) out[i] = 0.0f;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const float *partial = step->partials[rank];
        for (size_t i = 0; i < out_count; i++) {
            out[i] += partial[i];
        }
    }
    for (size_t i = 0; i < out_count; i++) {
        if (!mock_float_ptr_isfinite(&out[i])) {
            mock_error(err, err_size, "GLM 5.2 mock all-reduce output contains non-finite values");
            return false;
        }
    }
    return true;
}

bool ds4_glm52_mock_dcp_build_owner_ranges(
        uint64_t row_count,
        ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    if (!owners || row_count == 0) {
        mock_error(err, err_size, "GLM 5.2 mock DCP owner ranges require rows");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        owners[rank].rank = rank;
        owners[rank].start_row =
            (row_count * (uint64_t)rank) / DS4_GLM52_L0_RANK_COUNT;
        owners[rank].end_row =
            (row_count * (uint64_t)(rank + 1)) / DS4_GLM52_L0_RANK_COUNT;
        owners[rank].present = true;
    }
    return true;
}

uint64_t ds4_glm52_mock_dcp_row_checksum(int owner_rank,
                                         int layer,
                                         uint64_t token_step_j,
                                         uint64_t row_id) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_bytes(h, "glm52-mock/dcp-row");
    h = fnv1a_u64(h, (uint64_t)(uint32_t)owner_rank);
    h = fnv1a_u64(h, (uint64_t)(uint32_t)layer);
    h = fnv1a_u64(h, token_step_j);
    h = fnv1a_u64(h, row_id);
    return h;
}

static bool mock_dcp_validate_owner_ranges(
        const ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        uint64_t *total_rows,
        char *err,
        size_t err_size) {
    if (!owners || !total_rows) {
        mock_error(err, err_size, "GLM 5.2 mock DCP owner ranges are missing");
        return false;
    }
    uint64_t expected_start = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_mock_dcp_owner_range *o = &owners[rank];
        if (!o->present || o->rank != rank) {
            mock_error(err, err_size, "GLM 5.2 mock DCP owner rank is missing");
            return false;
        }
        if (o->start_row != expected_start || o->end_row < o->start_row) {
            mock_error(err, err_size, "GLM 5.2 mock DCP owner ranges have gaps or overlaps");
            return false;
        }
        expected_start = o->end_row;
    }
    if (expected_start == 0) {
        mock_error(err, err_size, "GLM 5.2 mock DCP owner ranges are empty");
        return false;
    }
    *total_rows = expected_start;
    return true;
}

static int mock_dcp_owner_for_row(
        const ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        uint64_t row_id) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (row_id >= owners[rank].start_row &&
            row_id < owners[rank].end_row) {
            return rank;
        }
    }
    return -1;
}

static void mock_dcp_sort_selection(ds4_glm52_mock_dcp_selection *sel) {
    for (int i = 1; i < sel->row_count; i++) {
        ds4_glm52_mock_dcp_row row = sel->rows[i];
        int j = i - 1;
        while (j >= 0 && sel->rows[j].row_id > row.row_id) {
            sel->rows[j + 1] = sel->rows[j];
            j--;
        }
        sel->rows[j + 1] = row;
    }
}

bool ds4_glm52_mock_dcp_exchange(
        const ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_dcp_row *rows,
        size_t row_count,
        const ds4_glm52_mock_dcp_request *requests,
        size_t request_count,
        ds4_glm52_mock_dcp_selection selections[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    uint64_t total_rows = 0;
    if (!mock_dcp_validate_owner_ranges(owners, &total_rows, err, err_size)) {
        return false;
    }
    if (!rows || row_count == 0 || !requests || request_count == 0 ||
        !selections) {
        mock_error(err, err_size, "GLM 5.2 mock DCP exchange requires rows and requests");
        return false;
    }
    if (request_count > DS4_GLM52_MOCK_DCP_MAX_ROWS *
            DS4_GLM52_L0_RANK_COUNT) {
        mock_error(err, err_size, "GLM 5.2 mock DCP exchange request set is too large");
        return false;
    }

    for (size_t i = 0; i < row_count; i++) {
        const ds4_glm52_mock_dcp_row *r = &rows[i];
        if (!r->present ||
            r->owner_rank < 0 ||
            r->owner_rank >= DS4_GLM52_L0_RANK_COUNT ||
            r->row_id >= total_rows ||
            mock_dcp_owner_for_row(owners, r->row_id) != r->owner_rank) {
            mock_error(err, err_size, "GLM 5.2 mock DCP row owner mapping is invalid");
            return false;
        }
        if (r->checksum != ds4_glm52_mock_dcp_row_checksum(
                r->owner_rank, r->layer, r->token_step_j, r->row_id)) {
            mock_error(err, err_size, "GLM 5.2 mock DCP row checksum mismatch");
            return false;
        }
        for (size_t j = 0; j < i; j++) {
            if (rows[j].present && rows[j].row_id == r->row_id) {
                mock_error(err, err_size, "GLM 5.2 mock DCP duplicate row payload");
                return false;
            }
        }
    }

    memset(selections, 0,
           sizeof(selections[0]) * DS4_GLM52_L0_RANK_COUNT);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        selections[rank].requester_rank = rank;
        selections[rank].present = true;
        selections[rank].checksum = fnv1a_u64(
            fnv1a_bytes(1469598103934665603ull,
                        "glm52-mock/dcp-selection"),
            (uint64_t)(uint32_t)rank);
    }

    for (size_t i = 0; i < request_count; i++) {
        const ds4_glm52_mock_dcp_request *req = &requests[i];
        if (!req->present ||
            req->requester_rank < 0 ||
            req->requester_rank >= DS4_GLM52_L0_RANK_COUNT ||
            req->row_id >= total_rows) {
            mock_error(err, err_size, "GLM 5.2 mock DCP request is invalid");
            return false;
        }
        const int owner_rank = mock_dcp_owner_for_row(owners, req->row_id);
        if (owner_rank < 0) {
            mock_error(err, err_size, "GLM 5.2 mock DCP selected row has no owner");
            return false;
        }

        const ds4_glm52_mock_dcp_row *found = NULL;
        for (size_t j = 0; j < row_count; j++) {
            if (rows[j].present && rows[j].row_id == req->row_id) {
                found = &rows[j];
                break;
            }
        }
        if (!found || found->owner_rank != owner_rank) {
            mock_error(err, err_size, "GLM 5.2 mock DCP selected row payload is missing");
            return false;
        }

        ds4_glm52_mock_dcp_selection *sel = &selections[req->requester_rank];
        bool duplicate_request = false;
        for (int k = 0; k < sel->row_count; k++) {
            if (sel->rows[k].row_id == req->row_id) {
                duplicate_request = true;
                break;
            }
        }
        if (duplicate_request) continue;
        if (sel->row_count >= DS4_GLM52_MOCK_DCP_MAX_ROWS) {
            mock_error(err, err_size, "GLM 5.2 mock DCP selected row output exceeds capacity");
            return false;
        }
        sel->rows[sel->row_count++] = *found;
    }

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_mock_dcp_selection *sel = &selections[rank];
        if (sel->row_count == 0) {
            mock_error(err, err_size, "GLM 5.2 mock DCP selected row output is empty");
            return false;
        }
        mock_dcp_sort_selection(sel);
        for (int i = 0; i < sel->row_count; i++) {
            sel->checksum = fnv1a_u64(sel->checksum, sel->rows[i].row_id);
            sel->checksum = fnv1a_u64(sel->checksum,
                                      (uint64_t)(uint32_t)sel->rows[i].owner_rank);
            sel->checksum = fnv1a_u64(sel->checksum, sel->rows[i].checksum);
        }
    }
    return true;
}

static bool mock_tp4_run_dcp_exchange_for_layer(
        const ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        int layer,
        ds4_glm52_mock_dcp_selection selections[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    ds4_glm52_mock_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT];
    if (!ds4_glm52_mock_dcp_build_owner_ranges(sessions[0].kv_length,
                                               owners,
                                               err,
                                               err_size)) {
        return false;
    }

    ds4_glm52_mock_dcp_request requests[
        DS4_GLM52_L0_RANK_COUNT * DS4_GLM52_MOCK_DCP_TOPK];
    ds4_glm52_mock_dcp_row rows[
        DS4_GLM52_L0_RANK_COUNT * DS4_GLM52_MOCK_DCP_TOPK];
    size_t request_count = 0;
    size_t row_count = 0;
    for (int requester = 0; requester < DS4_GLM52_L0_RANK_COUNT; requester++) {
        for (int k = 0; k < DS4_GLM52_MOCK_DCP_TOPK; k++) {
            int owner_rank = -1;
            for (int probe = 0; probe < DS4_GLM52_L0_RANK_COUNT; probe++) {
                const int candidate =
                    (requester + k + probe) % DS4_GLM52_L0_RANK_COUNT;
                if (owners[candidate].end_row > owners[candidate].start_row) {
                    owner_rank = candidate;
                    break;
                }
            }
            if (owner_rank < 0) {
                mock_error(err, err_size, "GLM 5.2 mock DCP selected row has no non-empty owner");
                return false;
            }
            const uint64_t span =
                owners[owner_rank].end_row - owners[owner_rank].start_row;
            const uint64_t offset =
                (sessions[requester].hidden_checksum +
                 sessions[0].token_step_j +
                 (uint64_t)(layer + 1) * 17u +
                 (uint64_t)k * 23u) % span;
            const uint64_t row_id = owners[owner_rank].start_row + offset;
            requests[request_count++] = (ds4_glm52_mock_dcp_request){
                .requester_rank = requester,
                .row_id = row_id,
                .present = true,
            };

            bool have_row = false;
            for (size_t i = 0; i < row_count; i++) {
                if (rows[i].row_id == row_id) {
                    have_row = true;
                    break;
                }
            }
            if (!have_row) {
                rows[row_count++] = (ds4_glm52_mock_dcp_row){
                    .owner_rank = owner_rank,
                    .layer = layer,
                    .token_step_j = sessions[0].token_step_j,
                    .row_id = row_id,
                    .checksum = ds4_glm52_mock_dcp_row_checksum(
                        owner_rank, layer, sessions[0].token_step_j, row_id),
                    .present = true,
                };
            }
        }
    }
    return ds4_glm52_mock_dcp_exchange(owners,
                                       rows,
                                       row_count,
                                       requests,
                                       request_count,
                                       selections,
                                       err,
                                       err_size);
}

static uint64_t mock_hash_f32_span(uint64_t h, const float *v, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, &v[i], sizeof(bits));
        h = fnv1a_u64(h, bits);
    }
    return h;
}

static void mock_fill_hidden_partial(
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_mock_allreduce_kind kind,
        int layer,
        float *out,
        size_t count) {
    uint64_t h = session->hidden_checksum;
    h = fnv1a_u64(h, model->model_hash);
    h = fnv1a_u64(h, session->token_step_j);
    h = fnv1a_u64(h, (uint64_t)(uint32_t)kind);
    h = fnv1a_u64(h, (uint64_t)(uint32_t)layer);
    h = fnv1a_u64(h, (uint64_t)(uint32_t)model->rank);
    const float scale = kind == DS4_GLM52_MOCK_ALLREDUCE_ATTN ?
        1.0f / 4096.0f : 1.0f / 3072.0f;
    for (size_t i = 0; i < count; i++) {
        h = fnv1a_u64(h, (uint64_t)i);
        const int centered = (int)(h % 2049u) - 1024;
        out[i] = (float)centered * scale;
    }
}

static bool mock_tp4_validate_collective_inputs(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_mock_model_validate(&models[rank], err, err_size)) {
            return false;
        }
        if (models[rank].rank != rank) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 collective model rank order is invalid");
            return false;
        }
        if (!sessions[rank].prefilled) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 collective requires prefilled rank sessions");
            return false;
        }
        if (models[rank].model_hash != models[0].model_hash ||
            sessions[rank].session_hash != sessions[0].session_hash ||
            sessions[rank].token_step_j != sessions[0].token_step_j ||
            sessions[rank].kv_length != sessions[0].kv_length) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 collective rank identity/cursor mismatch");
            return false;
        }
    }
    return true;
}

static bool mock_tp4_apply_one_allreduce(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_allreduce_kind kind,
        uint64_t seq,
        int layer,
        float *rank_partials,
        float *reduced,
        char *err,
        size_t err_size) {
    ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        float *p = rank_partials + (size_t)rank * DS4_GLM52_MOCK_N_EMBD;
        mock_fill_hidden_partial(&models[rank], &sessions[rank], kind,
                                 layer, p, DS4_GLM52_MOCK_N_EMBD);
        if (!ds4_glm52_mock_make_hidden_partial(&models[rank],
                                                &sessions[rank],
                                                kind,
                                                seq,
                                                layer,
                                                p,
                                                DS4_GLM52_MOCK_N_EMBD,
                                                &partials[rank],
                                                err,
                                                err_size)) {
            return false;
        }
    }

    ds4_glm52_mock_allreduce_step step = {0};
    const int order[] = {2, 0, 3, 1};
    for (int i = 0; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        if (!ds4_glm52_mock_allreduce_add_contribution(&step,
                                                       &partials[order[i]],
                                                       err,
                                                       err_size)) {
            return false;
        }
    }
    if (!ds4_glm52_mock_allreduce_finish(&step,
                                         reduced,
                                         DS4_GLM52_MOCK_N_EMBD,
                                         err,
                                         err_size)) {
        return false;
    }

    uint64_t h = fnv1a_u64(sessions[0].hidden_checksum,
                           (uint64_t)(uint32_t)kind);
    h = fnv1a_u64(h, (uint64_t)(uint32_t)layer);
    h = fnv1a_u64(h, seq);
    h = mock_hash_f32_span(h, reduced, DS4_GLM52_MOCK_N_EMBD);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        sessions[rank].hidden_checksum = h;
    }
    return true;
}

bool ds4_glm52_mock_tp4_apply_collectives(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    if (!mock_tp4_validate_collective_inputs(models, sessions, err, err_size)) {
        return false;
    }
    float *rank_partials = malloc(
        (size_t)DS4_GLM52_L0_RANK_COUNT *
        (size_t)DS4_GLM52_MOCK_N_EMBD *
        sizeof(float));
    float *reduced = malloc((size_t)DS4_GLM52_MOCK_N_EMBD * sizeof(float));
    if (!rank_partials || !reduced) {
        free(rank_partials);
        free(reduced);
        mock_error(err, err_size, "GLM 5.2 mock TP4 collective allocation failed");
        return false;
    }

    for (int layer = 0; layer < DS4_GLM52_MOCK_N_LAYER; layer++) {
        ds4_glm52_mock_dcp_selection selections[DS4_GLM52_L0_RANK_COUNT];
        if (!mock_tp4_run_dcp_exchange_for_layer(sessions,
                                                 layer,
                                                 selections,
                                                 err,
                                                 err_size)) {
            free(rank_partials);
            free(reduced);
            return false;
        }
        for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            sessions[rank].hidden_checksum =
                fnv1a_u64(sessions[rank].hidden_checksum,
                          selections[rank].checksum);
        }
        const uint64_t base_seq =
            sessions[0].token_step_j * UINT64_C(1000000) +
            (uint64_t)(layer + 1) * UINT64_C(10);
        if (!mock_tp4_apply_one_allreduce(models,
                                          sessions,
                                          DS4_GLM52_MOCK_ALLREDUCE_ATTN,
                                          base_seq + 1u,
                                          layer,
                                          rank_partials,
                                          reduced,
                                          err,
                                          err_size) ||
            !mock_tp4_apply_one_allreduce(models,
                                          sessions,
                                          DS4_GLM52_MOCK_ALLREDUCE_FFN,
                                          base_seq + 2u,
                                          layer,
                                          rank_partials,
                                          reduced,
                                          err,
                                          err_size)) {
            free(rank_partials);
            free(reduced);
            return false;
        }
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        mock_fill_logits(&models[rank], &sessions[rank], NULL, 0);
    }
    free(rank_partials);
    free(reduced);
    return true;
}

bool ds4_glm52_mock_tp4_collective_step(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_tp4_step *step,
        int last_token,
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    if (!models || !step || !sessions || !step->complete) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 collective step requires a complete rank step");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_mock_rank_step *r = &step->ranks[rank];
        if (!r->present || r->rank != rank) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 collective step rank is missing");
            return false;
        }
        memset(&sessions[rank], 0, sizeof(sessions[rank]));
        sessions[rank].session_hash = r->session_hash;
        sessions[rank].token_step_j = r->token_step_j;
        sessions[rank].kv_length = r->kv_length;
        sessions[rank].hidden_checksum = r->hidden_checksum;
        sessions[rank].logits_checksum = r->logits_checksum;
        sessions[rank].last_token = last_token;
        sessions[rank].prefilled = true;
    }
    return ds4_glm52_mock_tp4_apply_collectives(models,
                                                sessions,
                                                err,
                                                err_size);
}

bool ds4_glm52_mock_tp4_prefill(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const int *tokens,
        size_t token_count,
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        char *err,
        size_t err_size) {
    if (!models || !sessions) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 prefill requires rank models and sessions");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_mock_prefill(&models[rank],
                                    tokens,
                                    token_count,
                                    &sessions[rank],
                                    NULL,
                                    0,
                                    err,
                                    err_size)) {
            return false;
        }
    }
    return ds4_glm52_mock_tp4_apply_collectives(models, sessions, err, err_size);
}

bool ds4_glm52_mock_tp4_decode(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        int input_token,
        char *err,
        size_t err_size) {
    if (!models || !sessions) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 decode requires rank models and sessions");
        return false;
    }
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_mock_decode(&models[rank],
                                   &sessions[rank],
                                   input_token,
                                   NULL,
                                   0,
                                   err,
                                   err_size)) {
            return false;
        }
    }
    return ds4_glm52_mock_tp4_apply_collectives(models, sessions, err, err_size);
}

static ds4_glm52_l0_config mock_tp4_rank_cfg(int rank) {
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

static bool mock_tp4_rank_plans_tile(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        bool *q_heads_tile,
        bool *vocab_tiles) {
    bool q_seen[DS4_GLM52_MOCK_N_HEAD] = {false};
    int vocab_end = 0;
    bool q_ok = true;
    bool vocab_ok = true;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const int expected_q0 = rank * DS4_GLM52_L0_Q_HEADS_PER_RANK;
        const int expected_q1 = expected_q0 + DS4_GLM52_L0_Q_HEADS_PER_RANK;
        if (models[rank].rank != rank ||
            models[rank].q_head_start != expected_q0 ||
            models[rank].q_head_end != expected_q1) {
            q_ok = false;
        }
        for (int q = models[rank].q_head_start; q < models[rank].q_head_end; q++) {
            if (q < 0 || q >= DS4_GLM52_MOCK_N_HEAD || q_seen[q]) {
                q_ok = false;
            } else {
                q_seen[q] = true;
            }
        }
        if (models[rank].vocab_start != vocab_end ||
            models[rank].vocab_end <= models[rank].vocab_start) {
            vocab_ok = false;
        }
        vocab_end = models[rank].vocab_end;
    }
    for (int q = 0; q < DS4_GLM52_MOCK_N_HEAD; q++) {
        if (!q_seen[q]) q_ok = false;
    }
    if (vocab_end != DS4_GLM52_MOCK_N_VOCAB) vocab_ok = false;
    if (q_heads_tile) *q_heads_tile = q_ok;
    if (vocab_tiles) *vocab_tiles = vocab_ok;
    return q_ok && vocab_ok;
}

static bool mock_tp4_sessions_replicated(
        const ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT]) {
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!sessions[rank].prefilled ||
            sessions[rank].session_hash != sessions[0].session_hash ||
            sessions[rank].token_step_j != sessions[0].token_step_j ||
            sessions[rank].kv_length != sessions[0].kv_length ||
            sessions[rank].hidden_checksum != sessions[0].hidden_checksum) {
            return false;
        }
    }
    return sessions[0].prefilled;
}

bool ds4_glm52_mock_tp4_serve_request(
        const int *prompt_tokens,
        size_t prompt_token_count,
        ds4_glm52_mock_serve_observation *obs,
        char *err,
        size_t err_size) {
    if (!prompt_tokens || prompt_token_count == 0 || !obs) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve request requires prompt tokens and observation output");
        return false;
    }
    memset(obs, 0, sizeof(*obs));

    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_l0_config cfg = mock_tp4_rank_cfg(rank);
        if (!ds4_glm52_mock_model_init(&cfg,
                                       &models[rank],
                                       err,
                                       err_size)) {
            return false;
        }
    }

    obs->tp_size = DS4_GLM52_L0_TP_SIZE;
    obs->dcp_size = DS4_GLM52_L0_DCP_SIZE;
    obs->rank_count = DS4_GLM52_L0_RANK_COUNT;
    obs->model_hash = models[0].model_hash;
    mock_tp4_rank_plans_tile(models, &obs->q_heads_tile, &obs->vocab_tiles);
    if (!obs->q_heads_tile || !obs->vocab_tiles) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve rank plan does not tile");
        return false;
    }

    if (!ds4_glm52_mock_tp4_prefill(models,
                                    prompt_tokens,
                                    prompt_token_count,
                                    sessions,
                                    err,
                                    err_size)) {
        return false;
    }
    ds4_glm52_mock_tp4_step prefill_step = {0};
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_mock_tp4_step_add_rank(&prefill_step,
                                              &models[rank],
                                              &sessions[rank],
                                              DS4_GLM52_TP4_COMMAND_PREFILL,
                                              UINT64_C(1),
                                              err,
                                              err_size)) {
            return false;
        }
    }
    if (!prefill_step.complete) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve prefill gather is incomplete");
        return false;
    }
    obs->session_hash = sessions[0].session_hash;
    obs->prefill_token_step_j = sessions[0].token_step_j;
    obs->prefill_kv_length = sessions[0].kv_length;
    obs->prefill_hidden_checksum = sessions[0].hidden_checksum;
    obs->prefill_candidate_token = prefill_step.coordinator_token;
    obs->prefill_candidate_rank = prefill_step.coordinator_rank;
    obs->prefill_replicated = mock_tp4_sessions_replicated(sessions);
    if (!obs->prefill_replicated) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve prefill state is not replicated");
        return false;
    }
    if (prefill_step.coordinator_rank < 0 ||
        prefill_step.coordinator_rank >= DS4_GLM52_L0_RANK_COUNT) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve prefill winner rank is invalid");
        return false;
    }
    const ds4_glm52_mock_model *winner_model =
        &models[prefill_step.coordinator_rank];
    obs->coordinator_token_rank_owned =
        prefill_step.coordinator_token >= winner_model->vocab_start &&
        prefill_step.coordinator_token < winner_model->vocab_end;
    if (!obs->coordinator_token_rank_owned) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve prefill token is outside winner vocab shard");
        return false;
    }

    if (!ds4_glm52_mock_tp4_decode(models,
                                   sessions,
                                   prefill_step.coordinator_token,
                                   err,
                                   err_size)) {
        return false;
    }
    ds4_glm52_mock_tp4_step decode_step = {0};
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_mock_tp4_step_add_rank(&decode_step,
                                              &models[rank],
                                              &sessions[rank],
                                              DS4_GLM52_TP4_COMMAND_DECODE,
                                              UINT64_C(2),
                                              err,
                                              err_size)) {
            return false;
        }
    }
    if (!decode_step.complete) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve decode gather is incomplete");
        return false;
    }
    obs->decode_token_step_j = sessions[0].token_step_j;
    obs->decode_kv_length = sessions[0].kv_length;
    obs->decode_hidden_checksum = sessions[0].hidden_checksum;
    obs->logits_checksum = sessions[0].logits_checksum;
    obs->decode_candidate_token = decode_step.coordinator_token;
    obs->decode_candidate_rank = decode_step.coordinator_rank;
    obs->decode_replicated = mock_tp4_sessions_replicated(sessions);
    if (!obs->decode_replicated) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 serve decode state is not replicated");
        return false;
    }
    return true;
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t type;
    uint32_t payload_size;
} ds4_glm52_mock_step_frame_header;

typedef enum {
    DS4_GLM52_MOCK_FRAME_RANK_STEP = 1,
    DS4_GLM52_MOCK_FRAME_TOKENS = 2,
    DS4_GLM52_MOCK_FRAME_SESSION_STATE = 3,
} ds4_glm52_mock_frame_type;

typedef struct {
    int32_t rank;
    int32_t command;
    int32_t q_head_start;
    int32_t q_head_end;
    int32_t vocab_start;
    int32_t vocab_end;
    int32_t candidate_token;
    float candidate_score;
    uint64_t model_hash;
    uint64_t session_hash;
    uint64_t seq;
    uint64_t token_step_j;
    uint64_t kv_length;
    uint64_t hidden_checksum;
    uint64_t logits_checksum;
    uint32_t present;
} ds4_glm52_mock_step_payload;

typedef struct {
    uint32_t token_count;
    int32_t tokens[DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS];
} ds4_glm52_mock_tokens_payload;

typedef struct {
    uint64_t session_hash;
    uint64_t token_step_j;
    uint64_t kv_length;
    uint64_t hidden_checksum;
    uint64_t logits_checksum;
    int32_t last_token;
    uint32_t prefilled;
} ds4_glm52_mock_session_state_payload;

static size_t mock_tokens_payload_size(size_t token_count) {
    return sizeof(uint32_t) + token_count * sizeof(int32_t);
}

static bool mock_write_exact(int fd, const void *buf, size_t len) {
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

static bool mock_read_exact(int fd, void *buf, size_t len) {
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

bool ds4_glm52_mock_transport_send_rank_step(
        int fd,
        const ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size) {
    if (!rank_step || !rank_step->present) {
        mock_error(err, err_size, "GLM 5.2 mock transport rank step is missing");
        return false;
    }
    ds4_glm52_mock_step_payload payload;
    memset(&payload, 0, sizeof(payload));
    payload.rank = rank_step->rank;
    payload.command = rank_step->command;
    payload.q_head_start = rank_step->q_head_start;
    payload.q_head_end = rank_step->q_head_end;
    payload.vocab_start = rank_step->vocab_start;
    payload.vocab_end = rank_step->vocab_end;
    payload.candidate_token = rank_step->candidate_token;
    payload.candidate_score = rank_step->candidate_score;
    payload.model_hash = rank_step->model_hash;
    payload.session_hash = rank_step->session_hash;
    payload.seq = rank_step->seq;
    payload.token_step_j = rank_step->token_step_j;
    payload.kv_length = rank_step->kv_length;
    payload.hidden_checksum = rank_step->hidden_checksum;
    payload.logits_checksum = rank_step->logits_checksum;
    payload.present = rank_step->present ? 1u : 0u;

    ds4_glm52_mock_step_frame_header header = {
        .magic = DS4_GLM52_MOCK_STEP_MAGIC,
        .version = DS4_GLM52_MOCK_STEP_VERSION,
        .type = DS4_GLM52_MOCK_FRAME_RANK_STEP,
        .payload_size = (uint32_t)sizeof(payload),
    };
    if (!mock_write_exact(fd, &header, sizeof(header)) ||
        !mock_write_exact(fd, &payload, sizeof(payload))) {
        mock_error(err, err_size, "GLM 5.2 mock transport rank step write failed");
        return false;
    }
    return true;
}

bool ds4_glm52_mock_transport_recv_rank_step(
        int fd,
        ds4_glm52_mock_rank_step *rank_step,
        char *err,
        size_t err_size) {
    if (!rank_step) {
        mock_error(err, err_size, "GLM 5.2 mock transport rank step output is missing");
        return false;
    }
    ds4_glm52_mock_step_frame_header header;
    if (!mock_read_exact(fd, &header, sizeof(header))) {
        mock_error(err, err_size, "GLM 5.2 mock transport rank step header read failed");
        return false;
    }
    if (header.magic != DS4_GLM52_MOCK_STEP_MAGIC ||
        header.version != DS4_GLM52_MOCK_STEP_VERSION ||
        header.type != DS4_GLM52_MOCK_FRAME_RANK_STEP ||
        header.payload_size != sizeof(ds4_glm52_mock_step_payload)) {
        mock_error(err, err_size, "GLM 5.2 mock transport rank step frame mismatch");
        return false;
    }

    ds4_glm52_mock_step_payload payload;
    if (!mock_read_exact(fd, &payload, sizeof(payload))) {
        mock_error(err, err_size, "GLM 5.2 mock transport rank step payload read failed");
        return false;
    }
    memset(rank_step, 0, sizeof(*rank_step));
    rank_step->rank = payload.rank;
    rank_step->command = (ds4_glm52_tp4_command)payload.command;
    rank_step->q_head_start = payload.q_head_start;
    rank_step->q_head_end = payload.q_head_end;
    rank_step->vocab_start = payload.vocab_start;
    rank_step->vocab_end = payload.vocab_end;
    rank_step->candidate_token = payload.candidate_token;
    rank_step->candidate_score = payload.candidate_score;
    rank_step->model_hash = payload.model_hash;
    rank_step->session_hash = payload.session_hash;
    rank_step->seq = payload.seq;
    rank_step->token_step_j = payload.token_step_j;
    rank_step->kv_length = payload.kv_length;
    rank_step->hidden_checksum = payload.hidden_checksum;
    rank_step->logits_checksum = payload.logits_checksum;
    rank_step->present = payload.present != 0;
    return true;
}

bool ds4_glm52_mock_transport_send_tokens(
        int fd,
        const int *tokens,
        size_t token_count,
        char *err,
        size_t err_size) {
    if (token_count > DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS ||
        (token_count > 0 && !tokens)) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span is invalid");
        return false;
    }
    ds4_glm52_mock_tokens_payload payload;
    memset(&payload, 0, sizeof(payload));
    payload.token_count = (uint32_t)token_count;
    for (size_t i = 0; i < token_count; i++) {
        payload.tokens[i] = (int32_t)tokens[i];
    }
    const size_t payload_size = mock_tokens_payload_size(token_count);
    ds4_glm52_mock_step_frame_header header = {
        .magic = DS4_GLM52_MOCK_STEP_MAGIC,
        .version = DS4_GLM52_MOCK_STEP_VERSION,
        .type = DS4_GLM52_MOCK_FRAME_TOKENS,
        .payload_size = (uint32_t)payload_size,
    };
    if (!mock_write_exact(fd, &header, sizeof(header)) ||
        !mock_write_exact(fd, &payload, payload_size)) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span write failed");
        return false;
    }
    return true;
}

bool ds4_glm52_mock_transport_recv_tokens(
        int fd,
        int *tokens,
        size_t max_tokens,
        size_t *token_count,
        char *err,
        size_t err_size) {
    if (!tokens || !token_count) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span output is missing");
        return false;
    }
    *token_count = 0;
    ds4_glm52_mock_step_frame_header header;
    if (!mock_read_exact(fd, &header, sizeof(header))) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span header read failed");
        return false;
    }
    if (header.magic != DS4_GLM52_MOCK_STEP_MAGIC ||
        header.version != DS4_GLM52_MOCK_STEP_VERSION ||
        header.type != DS4_GLM52_MOCK_FRAME_TOKENS ||
        header.payload_size < sizeof(uint32_t) ||
        header.payload_size > sizeof(ds4_glm52_mock_tokens_payload)) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span frame mismatch");
        return false;
    }
    ds4_glm52_mock_tokens_payload payload;
    memset(&payload, 0, sizeof(payload));
    if (!mock_read_exact(fd, &payload.token_count, sizeof(payload.token_count))) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span payload read failed");
        return false;
    }
    const size_t expected_size = mock_tokens_payload_size(payload.token_count);
    if (expected_size != header.payload_size) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span payload size mismatch");
        return false;
    }
    if (payload.token_count > max_tokens ||
        payload.token_count > DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span exceeds receiver capacity");
        return false;
    }
    if (payload.token_count > 0 &&
        !mock_read_exact(fd, payload.tokens,
                         payload.token_count * sizeof(payload.tokens[0]))) {
        mock_error(err, err_size, "GLM 5.2 mock transport token span payload read failed");
        return false;
    }
    for (uint32_t i = 0; i < payload.token_count; i++) {
        tokens[i] = payload.tokens[i];
    }
    *token_count = payload.token_count;
    return true;
}

void ds4_glm52_mock_session_export(
        const ds4_glm52_mock_session *session,
        ds4_glm52_mock_session_state *state) {
    if (!session || !state) return;
    state->session_hash = session->session_hash;
    state->token_step_j = session->token_step_j;
    state->kv_length = session->kv_length;
    state->hidden_checksum = session->hidden_checksum;
    state->logits_checksum = session->logits_checksum;
    state->last_token = session->last_token;
    state->prefilled = session->prefilled;
}

void ds4_glm52_mock_session_import(
        ds4_glm52_mock_session *session,
        const ds4_glm52_mock_session_state *state) {
    if (!session || !state) return;
    session->session_hash = state->session_hash;
    session->token_step_j = state->token_step_j;
    session->kv_length = state->kv_length;
    session->hidden_checksum = state->hidden_checksum;
    session->logits_checksum = state->logits_checksum;
    session->last_token = state->last_token;
    session->prefilled = state->prefilled;
}

bool ds4_glm52_mock_transport_send_session_state(
        int fd,
        const ds4_glm52_mock_session_state *state,
        char *err,
        size_t err_size) {
    if (!state || !state->prefilled) {
        mock_error(err, err_size, "GLM 5.2 mock transport session state is missing");
        return false;
    }
    ds4_glm52_mock_session_state_payload payload;
    memset(&payload, 0, sizeof(payload));
    payload.session_hash = state->session_hash;
    payload.token_step_j = state->token_step_j;
    payload.kv_length = state->kv_length;
    payload.hidden_checksum = state->hidden_checksum;
    payload.logits_checksum = state->logits_checksum;
    payload.last_token = state->last_token;
    payload.prefilled = state->prefilled ? 1u : 0u;

    ds4_glm52_mock_step_frame_header header = {
        .magic = DS4_GLM52_MOCK_STEP_MAGIC,
        .version = DS4_GLM52_MOCK_STEP_VERSION,
        .type = DS4_GLM52_MOCK_FRAME_SESSION_STATE,
        .payload_size = (uint32_t)sizeof(payload),
    };
    if (!mock_write_exact(fd, &header, sizeof(header)) ||
        !mock_write_exact(fd, &payload, sizeof(payload))) {
        mock_error(err, err_size, "GLM 5.2 mock transport session state write failed");
        return false;
    }
    return true;
}

bool ds4_glm52_mock_transport_recv_session_state(
        int fd,
        ds4_glm52_mock_session_state *state,
        char *err,
        size_t err_size) {
    if (!state) {
        mock_error(err, err_size, "GLM 5.2 mock transport session state output is missing");
        return false;
    }
    ds4_glm52_mock_step_frame_header header;
    if (!mock_read_exact(fd, &header, sizeof(header))) {
        mock_error(err, err_size, "GLM 5.2 mock transport session state header read failed");
        return false;
    }
    if (header.magic != DS4_GLM52_MOCK_STEP_MAGIC ||
        header.version != DS4_GLM52_MOCK_STEP_VERSION ||
        header.type != DS4_GLM52_MOCK_FRAME_SESSION_STATE ||
        header.payload_size != sizeof(ds4_glm52_mock_session_state_payload)) {
        mock_error(err, err_size, "GLM 5.2 mock transport session state frame mismatch");
        return false;
    }

    ds4_glm52_mock_session_state_payload payload;
    if (!mock_read_exact(fd, &payload, sizeof(payload))) {
        mock_error(err, err_size, "GLM 5.2 mock transport session state payload read failed");
        return false;
    }
    memset(state, 0, sizeof(*state));
    state->session_hash = payload.session_hash;
    state->token_step_j = payload.token_step_j;
    state->kv_length = payload.kv_length;
    state->hidden_checksum = payload.hidden_checksum;
    state->logits_checksum = payload.logits_checksum;
    state->last_token = payload.last_token;
    state->prefilled = payload.prefilled != 0;
    return true;
}

void ds4_glm52_mock_tokenize(const char *text, ds4_tokens *out) {
    if (!out) return;
    if (!text) text = "";
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') continue;
        ds4_tokens_push(out, 4096 + (int)*p);
    }
    if (out->len == 0) ds4_tokens_push(out, 4096);
}

char *ds4_glm52_mock_token_text(int token, size_t *len) {
    char buf[32];
    snprintf(buf, sizeof(buf), "<m%d>", token);
    const size_t n = strlen(buf);
    char *out = malloc(n + 1u);
    if (!out) return NULL;
    memcpy(out, buf, n + 1u);
    if (len) *len = n;
    return out;
}
