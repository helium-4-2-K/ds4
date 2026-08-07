#include "ds4_glm52_mock.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (!logits || logits_count == 0) return;
    for (size_t i = 0; i < logits_count; i++) logits[i] = -120.0f;
    uint64_t h = session->hidden_checksum;
    h = fnv1a_u64(h, model ? (uint64_t)model->rank : 0);
    h = fnv1a_u64(h, session->token_step_j);
    const size_t vocab = logits_count < DS4_GLM52_MOCK_N_VOCAB ?
        logits_count : DS4_GLM52_MOCK_N_VOCAB;
    const int best = (int)(h % (uint64_t)vocab);
    logits[best] = 12.0f;
    for (int i = 1; i <= 4; i++) {
        logits[(best + i) % (int)vocab] = 7.0f - (float)i;
    }
    session->logits_checksum = h;
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
    session->token_step_j = token_count - 1u;
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
                                        session->token_step_j + 1u);
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

bool ds4_glm52_mock_tp4_step_add_rank(
        ds4_glm52_mock_tp4_step *step,
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        ds4_glm52_tp4_command command,
        uint64_t seq,
        char *err,
        size_t err_size) {
    if (!step || !session || !session->prefilled) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 step requires rank session state");
        return false;
    }
    if (command != DS4_GLM52_TP4_COMMAND_PREFILL &&
        command != DS4_GLM52_TP4_COMMAND_DECODE) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 step only accepts prefill/decode");
        return false;
    }
    if (seq == 0) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 step requires nonzero command sequence");
        return false;
    }
    if (!ds4_glm52_mock_model_validate(model, err, err_size)) return false;
    const int rank = model->rank;
    if (step->rank_mask == 0) {
        memset(step, 0, sizeof(*step));
        step->command = command;
        step->seq = seq;
        step->session_hash = session->session_hash;
        step->token_step_j = session->token_step_j;
        step->kv_length = session->kv_length;
        step->coordinator_token = -1;
        step->coordinator_rank = -1;
        step->coordinator_score = -1.0e30f;
    } else {
        if (step->command != command) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 command mismatch");
            return false;
        }
        if (step->seq != seq) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 sequence mismatch");
            return false;
        }
        if (step->session_hash != session->session_hash) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 session identity mismatch");
            return false;
        }
        if (step->token_step_j != session->token_step_j ||
            step->kv_length != session->kv_length) {
            mock_error(err, err_size, "GLM 5.2 mock TP4 cursor mismatch");
            return false;
        }
    }
    if (step->rank_mask & (1u << rank)) {
        mock_error(err, err_size, "GLM 5.2 mock TP4 duplicate rank contribution");
        return false;
    }

    int candidate = -1;
    float score = 0.0f;
    if (!ds4_glm52_mock_rank_vocab_candidate(model, session, &candidate,
                                             &score, err, err_size)) {
        return false;
    }

    ds4_glm52_mock_rank_step *r = &step->ranks[rank];
    memset(r, 0, sizeof(*r));
    r->rank = rank;
    r->q_head_start = model->q_head_start;
    r->q_head_end = model->q_head_end;
    r->vocab_start = model->vocab_start;
    r->vocab_end = model->vocab_end;
    r->candidate_token = candidate;
    r->candidate_score = score;
    r->token_step_j = session->token_step_j;
    r->kv_length = session->kv_length;
    r->hidden_checksum = session->hidden_checksum;
    r->logits_checksum = session->logits_checksum;
    r->present = true;
    step->rank_mask |= 1u << rank;
    if (score > step->coordinator_score) {
        step->coordinator_score = score;
        step->coordinator_token = candidate;
        step->coordinator_rank = rank;
    }
    return mock_tp4_check_complete(step, err, err_size);
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
