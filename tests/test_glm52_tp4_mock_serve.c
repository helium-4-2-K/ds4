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
    check(obs.request_bound, "serve smoke should bind the request session");
    check(obs.prefill_enqueued, "serve smoke should enqueue prefill");

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
    check(obs.generated_count == 1,
          "legacy serve helper should generate one token");
    check(obs.stream_event_count == obs.generated_count,
          "legacy serve helper should stream one event per generated token");
    check(obs.generated_tokens[0] == obs.decode_candidate_token,
          "legacy serve helper should record the streamed decode token");
    check(obs.generated_token_ranks[0] == obs.decode_candidate_rank,
          "legacy serve helper should record the streamed token owner rank");
    check(obs.stopped_by_max_tokens,
          "legacy serve helper should stop at its one-token budget");
}

static void test_mock_serve_multi_token_stream(void) {
    const int prompt[] = {200, 201, 202, 203};
    ds4_glm52_mock_serve_observation obs;
    ds4_glm52_mock_serve_request req = {
        .prompt_tokens = prompt,
        .prompt_token_count = sizeof(prompt) / sizeof(prompt[0]),
        .max_decode_tokens = 4,
        .stop_token = -1,
        .fail_kind = DS4_GLM52_MOCK_SERVE_FAIL_NONE,
        .fail_decode_index = 0,
    };
    char err[192] = "";

    check(ds4_glm52_mock_tp4_serve_request_ex(&req, &obs, err, sizeof(err)),
          "multi-token mock serve request should complete");
    check(obs.request_bound, "multi-token request should bind session");
    check(obs.prefill_enqueued, "multi-token request should enqueue prefill");
    check(obs.generated_count == 4,
          "multi-token request should generate the requested budget");
    check(obs.stream_event_count == obs.generated_count,
          "multi-token request should stream every generated token");
    check(obs.decode_token_step_j == obs.prefill_token_step_j + 4u,
          "multi-token request should advance cursor once per token");
    check(obs.decode_kv_length == obs.prefill_kv_length + 4u,
          "multi-token request should append one KV row per token");
    check(obs.stopped_by_max_tokens,
          "multi-token request should stop at max token budget");
    check(!obs.stopped_by_stop_token,
          "multi-token request should not claim stop-token termination");
    for (int i = 0; i < obs.generated_count; i++) {
        check(obs.generated_tokens[i] >= 0 &&
                  obs.generated_tokens[i] < DS4_GLM52_MOCK_N_VOCAB,
              "streamed token should be inside global vocab");
        check(obs.generated_token_ranks[i] >= 0 &&
                  obs.generated_token_ranks[i] < DS4_GLM52_L0_RANK_COUNT,
              "streamed token owner should be a TP rank");
    }
}

static void test_mock_serve_stop_token_policy(void) {
    const int prompt[] = {300, 301, 302};
    ds4_glm52_mock_serve_observation first;
    ds4_glm52_mock_serve_request one = {
        .prompt_tokens = prompt,
        .prompt_token_count = sizeof(prompt) / sizeof(prompt[0]),
        .max_decode_tokens = 1,
        .stop_token = -1,
        .fail_kind = DS4_GLM52_MOCK_SERVE_FAIL_NONE,
        .fail_decode_index = 0,
    };
    char err[192] = "";
    check(ds4_glm52_mock_tp4_serve_request_ex(&one, &first, err, sizeof(err)),
          "one-token request should reveal deterministic stop token");

    ds4_glm52_mock_serve_observation obs;
    ds4_glm52_mock_serve_request stop = {
        .prompt_tokens = prompt,
        .prompt_token_count = sizeof(prompt) / sizeof(prompt[0]),
        .max_decode_tokens = 4,
        .stop_token = first.generated_tokens[0],
        .fail_kind = DS4_GLM52_MOCK_SERVE_FAIL_NONE,
        .fail_decode_index = 0,
    };
    check(ds4_glm52_mock_tp4_serve_request_ex(&stop, &obs, err, sizeof(err)),
          "stop-token request should complete");
    check(obs.generated_count == 1,
          "stop-token request should stop after first matching token");
    check(obs.stream_event_count == 1,
          "stop-token request should stream the matching token");
    check(obs.stopped_by_stop_token,
          "stop-token request should claim stop-token termination");
    check(!obs.stopped_by_max_tokens,
          "stop-token request should not claim max-token termination");
}

static void check_failure_preserves_cursor(
        ds4_glm52_mock_serve_failure_kind kind,
        const char *label) {
    const int prompt[] = {400, 401, 402, 403};
    ds4_glm52_mock_serve_observation obs;
    ds4_glm52_mock_serve_request req = {
        .prompt_tokens = prompt,
        .prompt_token_count = sizeof(prompt) / sizeof(prompt[0]),
        .max_decode_tokens = 3,
        .stop_token = -1,
        .fail_kind = kind,
        .fail_decode_index = 1,
    };
    char err[192] = "";

    check(!ds4_glm52_mock_tp4_serve_request_ex(&req, &obs, err, sizeof(err)),
          label);
    check(obs.failure_kind == (int)kind,
          "failure observation should record the injected kind");
    check(obs.failure_decode_index == 1,
          "failure observation should record decode index");
    check(obs.generated_count == 1,
          "failure at decode index 1 should stream exactly one token first");
    check(obs.stream_event_count == 1,
          "failure should not stream the failed token");
    check(obs.failure_token_step_j == obs.prefill_token_step_j + 1u,
          "failure should observe cursor before failed decode commit");
    check(obs.failure_kv_length == obs.prefill_kv_length + 1u,
          "failure should observe KV length before failed decode commit");
    check(obs.decode_token_step_j == obs.failure_token_step_j,
          "failure should preserve committed decode cursor");
    check(obs.decode_kv_length == obs.failure_kv_length,
          "failure should preserve committed KV length");
    check(obs.failed_without_cursor_advance,
          "failure should explicitly prove no failed-token cursor advance");
}

static void test_mock_serve_failure_paths_preserve_state(void) {
    check_failure_preserves_cursor(DS4_GLM52_MOCK_SERVE_FAIL_WORKER_LOSS,
                                   "worker-loss injection should fail");
    check_failure_preserves_cursor(DS4_GLM52_MOCK_SERVE_FAIL_BAD_CONTRIBUTION,
                                   "bad-contribution injection should fail");
    check_failure_preserves_cursor(DS4_GLM52_MOCK_SERVE_FAIL_CURSOR_DIVERGENCE,
                                   "cursor-divergence injection should fail");
    check_failure_preserves_cursor(DS4_GLM52_MOCK_SERVE_FAIL_CANCEL,
                                   "cancel injection should fail");
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
    ds4_glm52_mock_serve_request bad_budget = {
        .prompt_tokens = (const int[]){1},
        .prompt_token_count = 1,
        .max_decode_tokens = 0,
        .stop_token = -1,
        .fail_kind = DS4_GLM52_MOCK_SERVE_FAIL_NONE,
        .fail_decode_index = 0,
    };
    check(!ds4_glm52_mock_tp4_serve_request_ex(&bad_budget, &obs,
                                               err, sizeof(err)),
          "serve request should reject zero decode budget");
    bad_budget.max_decode_tokens = DS4_GLM52_MOCK_MAX_GENERATED_TOKENS + 1u;
    check(!ds4_glm52_mock_tp4_serve_request_ex(&bad_budget, &obs,
                                               err, sizeof(err)),
          "serve request should reject excessive decode budget");
    bad_budget.max_decode_tokens = 1;
    bad_budget.fail_kind = (ds4_glm52_mock_serve_failure_kind)99;
    check(!ds4_glm52_mock_tp4_serve_request_ex(&bad_budget, &obs,
                                               err, sizeof(err)),
          "serve request should reject invalid failure injection kind");
}

int main(void) {
    test_mock_serve_prefill_decode_observation();
    test_mock_serve_multi_token_stream();
    test_mock_serve_stop_token_policy();
    test_mock_serve_failure_paths_preserve_state();
    test_mock_serve_rejects_invalid_request();
    puts("test_glm52_tp4_mock_serve: ok");
    return 0;
}
