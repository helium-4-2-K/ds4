#include "ds4.h"
#include "ds4_glm52_mock.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_mock: %s\n", msg);
        exit(1);
    }
}

static ds4_engine_options mock_options(void) {
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = "/mock/glm-5.2";
    opt.context_size = 64;
    opt.backend = DS4_BACKEND_CPU;
    opt.mtp_draft_tokens = 1;
    opt.mtp_margin = 3.0f;
    opt.glm52_tp4_l0 = true;
    opt.glm52_tp4_mock_model = true;
    opt.glm52_tp4_mock_matmul = true;
    opt.glm52_tp4_rank_set = true;
    opt.glm52_tp4_rank = 2;
    opt.glm52_tp4_rank_plan = "/mock/rank-plan.textproto";
    opt.glm52_tp4_fabric_addr = "127.0.0.1:40002";
    return opt;
}

static void test_mock_descriptor_shape(void) {
    ds4_engine_options opt = mock_options();
    ds4_glm52_l0_config cfg;
    char err[192] = "";
    check(ds4_glm52_l0_config_from_engine(&opt, &cfg, err, sizeof(err)),
          "mock config should convert from engine options");
    check(ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "mock config should validate");

    ds4_glm52_mock_model model;
    check(ds4_glm52_mock_model_init(&cfg, &model, err, sizeof(err)),
          "mock model should initialize");
    check(model.rank == 2, "rank should be preserved");
    check(model.q_head_start == 32 && model.q_head_end == 48,
          "rank 2 should own Q heads [32,48)");
    check(model.vocab_start == DS4_GLM52_MOCK_N_VOCAB / 2,
          "rank 2 should own third vocab shard start");
    check(model.tensor_count == 3 + DS4_GLM52_MOCK_N_LAYER * 2,
          "mock descriptor should include replicated, vocab, and per-layer Q/MLA tensors");

    bool saw_q = false;
    bool saw_mla = false;
    bool saw_vocab = false;
    for (int i = 0; i < model.tensor_count; i++) {
        const ds4_glm52_mock_tensor_desc *t = &model.tensors[i];
        if (t->role == DS4_GLM52_LAYOUT_ROLE_Q_HEAD &&
            t->owner_rank == 2 &&
            t->range_start == 32 &&
            t->range_end == 48) {
            saw_q = true;
        }
        if (t->role == DS4_GLM52_LAYOUT_ROLE_MLA_KV &&
            t->scope == DS4_GLM52_LAYOUT_SCOPE_REPLICATED &&
            t->range_end == DS4_GLM52_MOCK_N_KV_LORA +
                            DS4_GLM52_MOCK_N_QK_ROPE) {
            saw_mla = true;
        }
        if (t->role == DS4_GLM52_LAYOUT_ROLE_VOCAB &&
            t->owner_rank == 2 &&
            t->range_start == model.vocab_start &&
            t->range_end == model.vocab_end) {
            saw_vocab = true;
        }
    }
    check(saw_q, "descriptor should expose rank-local Q-head ownership");
    check(saw_mla, "descriptor should expose replicated MLA KV projection");
    check(saw_vocab, "descriptor should expose rank-local vocab shard");
}

static void test_mock_engine_session_round_trip(void) {
    ds4_engine_options opt = mock_options();
    ds4_engine *engine = NULL;
    check(ds4_engine_open(&engine, &opt) == 0, "mock engine should open");
    check(engine != NULL, "mock engine pointer should be returned");
    check(ds4_engine_vocab_size(engine) == DS4_GLM52_MOCK_N_VOCAB,
          "mock engine should report full GLM 5.2 vocab size");
    check(ds4_engine_layer_count(engine) == DS4_GLM52_MOCK_N_LAYER -
                                      DS4_GLM52_MOCK_N_NEXTN,
          "mock engine should report GLM 5.2 normal layer count");

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, "mock glm prompt", &prompt);
    check(prompt.len > 0, "mock tokenizer should produce prompt tokens");
    ds4_chat_append_message(engine, &prompt, "assistant", "prior answer");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    for (int i = 0; i < prompt.len; i++) {
        check(prompt.v[i] >= 0 && prompt.v[i] < DS4_GLM52_MOCK_N_VOCAB,
              "mock chat token APIs should not append negative or out-of-range ids");
    }

    ds4_session *session = NULL;
    check(ds4_session_create(&session, engine, 64) == 0,
          "mock session should be created");
    char err[192] = "";
    check(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0,
          "mock session sync should prefill");
    check(ds4_session_pos(session) == prompt.len,
          "mock prefill should advance session position");

    int first = ds4_session_argmax(session);
    check(first >= 0 && first < DS4_GLM52_MOCK_N_VOCAB,
          "mock argmax after prefill should be in vocab range");
    check(ds4_session_eval(session, first, err, sizeof(err)) == 0,
          "mock decode should evaluate one token");
    check(ds4_session_pos(session) == prompt.len + 1,
          "mock decode should advance by one token");
    int second = ds4_session_argmax(session);
    check(second >= 0 && second < DS4_GLM52_MOCK_N_VOCAB,
          "mock argmax after decode should be in vocab range");

    size_t piece_len = 0;
    char *piece = ds4_token_text(engine, second, &piece_len);
    check(piece && piece_len > 0 && strstr(piece, "<m") == piece,
          "mock token text should render visibly");
    free(piece);

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
}

static void test_mock_requires_mock_matmul(void) {
    ds4_engine_options opt = mock_options();
    opt.glm52_tp4_mock_matmul = false;
    ds4_engine *engine = NULL;
    check(ds4_engine_open(&engine, &opt) != 0,
          "mock model should fail closed without mock matmul");
    check(engine == NULL, "failed mock engine open should not return engine");
}

int main(void) {
    test_mock_descriptor_shape();
    test_mock_requires_mock_matmul();
    test_mock_engine_session_round_trip();
    puts("test_glm52_mock: ok");
    return 0;
}
