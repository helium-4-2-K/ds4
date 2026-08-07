#include "ds4_glm52_mock.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_tp4_mock_serve: %s\n", msg);
        exit(1);
    }
}

static void test_mock_serve_prefill_decode_observation(void) {
    const int prompt[] = {100, 101, 102, 103, 104};
    ds4_glm52_mock_serve_observation obs;
    char err[192] = "";

    if (!ds4_glm52_mock_tp4_serve_request(prompt,
                                          sizeof(prompt) / sizeof(prompt[0]),
                                          &obs,
                                          err,
                                          sizeof(err))) {
        fprintf(stderr, "test_glm52_tp4_mock_serve: %s\n", err);
        check(false, "mock TP4 serve request should run prefill and one decode");
    }

    check(obs.tp_size == DS4_GLM52_L0_TP_SIZE,
          "serve smoke should run with TP4");
    check(obs.dcp_size == DS4_GLM52_L0_DCP_SIZE,
          "serve smoke should run with DCP4");
    check(obs.rank_count == DS4_GLM52_L0_RANK_COUNT,
          "serve smoke should cover four ranks");
    check(obs.model_hash != 0, "serve smoke should bind one model identity");
    check(obs.session_hash != 0, "serve smoke should bind one session identity");
    check(obs.q_heads_tile, "Q heads should tile [0,64) exactly once");
    check(obs.vocab_tiles, "vocab shards should tile global vocab exactly once");
    check(obs.prefill_replicated,
          "prefill should publish replicated hidden/cursor state");
    check(obs.decode_replicated,
          "decode should publish replicated hidden/cursor state");
    check(obs.coordinator_token_rank_owned,
          "prefill-selected token should come from the winning vocab shard");

    check(obs.prefill_token_step_j == sizeof(prompt) / sizeof(prompt[0]),
          "prefill should advance token cursor to prompt length");
    check(obs.prefill_kv_length == sizeof(prompt) / sizeof(prompt[0]),
          "prefill should commit prompt KV length");
    check(obs.decode_token_step_j == obs.prefill_token_step_j + 1u,
          "decode should advance token cursor by one");
    check(obs.decode_kv_length == obs.prefill_kv_length + 1u,
          "decode should append one KV row");
    check(obs.prefill_hidden_checksum != 0 &&
              obs.decode_hidden_checksum != 0 &&
              obs.prefill_hidden_checksum != obs.decode_hidden_checksum,
          "decode hidden should reflect the accepted generated token");
    check(obs.logits_checksum != 0,
          "decode should publish a rank-local logits summary for top-k gather");
    check(obs.prefill_candidate_rank >= 0 &&
              obs.prefill_candidate_rank < DS4_GLM52_L0_RANK_COUNT,
          "prefill candidate rank should be one of the TP4 ranks");
    check(obs.decode_candidate_rank >= 0 &&
              obs.decode_candidate_rank < DS4_GLM52_L0_RANK_COUNT,
          "decode candidate rank should be one of the TP4 ranks");
    check(obs.prefill_candidate_token >= 0 &&
              obs.prefill_candidate_token < DS4_GLM52_MOCK_N_VOCAB,
          "prefill candidate token should be in global vocab");
    check(obs.decode_candidate_token >= 0 &&
              obs.decode_candidate_token < DS4_GLM52_MOCK_N_VOCAB,
          "decode candidate token should be in global vocab");
}

static void test_mock_serve_rejects_invalid_request(void) {
    ds4_glm52_mock_serve_observation obs;
    char err[192] = "";

    check(!ds4_glm52_mock_tp4_serve_request(NULL, 1, &obs,
                                            err, sizeof(err)),
          "serve request should reject missing prompt storage");
    check(!ds4_glm52_mock_tp4_serve_request((const int[]){1}, 0, &obs,
                                            err, sizeof(err)),
          "serve request should reject empty prompt");
    check(!ds4_glm52_mock_tp4_serve_request((const int[]){1}, 1, NULL,
                                            err, sizeof(err)),
          "serve request should reject missing observation output");
}

int main(void) {
    test_mock_serve_prefill_decode_observation();
    test_mock_serve_rejects_invalid_request();
    puts("test_glm52_tp4_mock_serve: ok");
    return 0;
}
