#include "ds4.h"
#include "ds4_glm52_dcp_mock.h"
#include "ds4_glm52_l0.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_dcp_row_exchange: %s\n", msg);
        exit(1);
    }
}

#define PER_RANK_ROWS 8
#define TOTAL_L0_ROWS (DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS)
#define KV_PAYLOAD_BYTES 8
#define K_ROPE_PAYLOAD_BYTES 6

/* Canonical ownership plan: rank r owns [r*8, r*8+8) over rows 0..31. */
static void build_plan(ds4_glm52_dcp_plan *plan) {
    ds4_glm52_dcp_plan_init(plan);
    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        ds4_glm52_dcp_plan_add_span(plan, r,
                                    r * PER_RANK_ROWS,
                                    (r + 1) * PER_RANK_ROWS);
    }
}

/* Full catalog: one row per position 0..31, owner = owning rank, with a
 * deterministic payload digest derived from (row_id, owner). */
static size_t build_catalog(ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS]) {
    size_t n = 0;
    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        for (int rid = r * PER_RANK_ROWS; rid < (r + 1) * PER_RANK_ROWS; rid++) {
            ds4_glm52_dcp_row row;
            ds4_glm52_dcp_row_init(&row);
            row.row_id = rid;
            row.layer_index = 0;
            row.position = rid;
            row.dcp_owner = r;
            row.kv_hash = (uint64_t)(rid * 131u + r * 7u) * 1099511628211ull;
            row.k_rope_hash = (uint64_t)(rid + 1u) * 2654435761u;
            row.present = true;
            rows[n++] = row;
        }
    }
    return n;
}

static void request_rank(ds4_glm52_dcp_request *req, int rank, const int *rows, int count) {
    memset(req, 0, sizeof(*req));
    req->requester_rank = rank;
    req->selection_count = count;
    for (int i = 0; i < count; i++) req->selected_rows[i] = rows[i];
    req->present = true;
}

static uint32_t full_rank_mask(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

static void build_l0_dcp_requests(
        ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT],
        const size_t selection_counts[DS4_GLM52_L0_RANK_COUNT]) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_dcp_exchange_request *req = &requests[rank];
        memset(req, 0, sizeof(*req));
        req->rank = rank;
        req->dcp_size = DS4_GLM52_L0_DCP_SIZE;
        req->rank_count = DS4_GLM52_L0_RANK_COUNT;
        req->layer_index = 3;
        req->seq = 41;
        req->model_hash = 0x52000010u;
        req->session_hash = 0x52000011u;
        req->token_step_j = 29;
        req->selected_row_count = (int)selection_counts[rank];
        req->owner_rank_mask = full_rank_mask();
        req->ownership_plan_valid = true;
        req->append_ordered_kv = true;
        req->row_payload_ready = true;
        req->transport_ready = true;
    }
}

static size_t build_l0_dcp_catalog(
        ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_dcp_row_payload catalog[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS]) {
    size_t n = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        owners[rank].rank = rank;
        owners[rank].row_start = (uint64_t)(rank * PER_RANK_ROWS);
        owners[rank].row_end = (uint64_t)((rank + 1) * PER_RANK_ROWS);
        owners[rank].present = true;
        for (int row_id = rank * PER_RANK_ROWS;
             row_id < (rank + 1) * PER_RANK_ROWS;
             row_id++) {
            catalog[n].row_id = (uint64_t)row_id;
            catalog[n].owner_rank = rank;
            catalog[n].layer_index = 3;
            catalog[n].token_step_j = (uint64_t)row_id;
            catalog[n].kv_hash =
                (uint64_t)(row_id + 1) * UINT64_C(1099511628211);
            catalog[n].k_rope_hash =
                (uint64_t)(row_id + 3) * UINT64_C(2654435761);
            catalog[n].present = true;
            n++;
        }
    }
    return n;
}

static size_t build_l0_dcp_bound_catalog(
        ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_dcp_bound_row_payload catalog[TOTAL_L0_ROWS],
        unsigned char kv_storage[TOTAL_L0_ROWS][KV_PAYLOAD_BYTES],
        unsigned char k_rope_storage[TOTAL_L0_ROWS][K_ROPE_PAYLOAD_BYTES]) {
    size_t n = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        owners[rank].rank = rank;
        owners[rank].row_start = (uint64_t)(rank * PER_RANK_ROWS);
        owners[rank].row_end = (uint64_t)((rank + 1) * PER_RANK_ROWS);
        owners[rank].present = true;
        for (int row_id = rank * PER_RANK_ROWS;
             row_id < (rank + 1) * PER_RANK_ROWS;
             row_id++) {
            for (size_t i = 0; i < KV_PAYLOAD_BYTES; i++) {
                kv_storage[n][i] = (unsigned char)(row_id * 17 + (int)i);
            }
            for (size_t i = 0; i < K_ROPE_PAYLOAD_BYTES; i++) {
                k_rope_storage[n][i] =
                    (unsigned char)(row_id * 31 + rank + (int)i);
            }

            memset(&catalog[n], 0, sizeof(catalog[n]));
            catalog[n].row.row_id = (uint64_t)row_id;
            catalog[n].row.owner_rank = rank;
            catalog[n].row.layer_index = 3;
            catalog[n].row.token_step_j = (uint64_t)row_id;
            catalog[n].row.kv_hash =
                ds4_glm52_dcp_payload_hash_host(kv_storage[n],
                                                KV_PAYLOAD_BYTES);
            catalog[n].row.k_rope_hash =
                ds4_glm52_dcp_payload_hash_host(k_rope_storage[n],
                                                K_ROPE_PAYLOAD_BYTES);
            catalog[n].row.present = true;
            catalog[n].kv_handle = kv_storage[n];
            catalog[n].kv_byte_offset = 0;
            catalog[n].kv_byte_count = KV_PAYLOAD_BYTES;
            catalog[n].kv_capacity_bytes = KV_PAYLOAD_BYTES;
            catalog[n].k_rope_handle = k_rope_storage[n];
            catalog[n].k_rope_byte_offset = 0;
            catalog[n].k_rope_byte_count = K_ROPE_PAYLOAD_BYTES;
            catalog[n].k_rope_capacity_bytes = K_ROPE_PAYLOAD_BYTES;
            catalog[n].ready = true;
            n++;
        }
    }
    return n;
}

static void test_l0_dcp_selected_rows_host(void) {
    ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_dcp_row_payload catalog[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t catalog_count = build_l0_dcp_catalog(owners, catalog);
    const uint64_t selected0[] = {28, 5, 20, 2};
    const uint64_t selected1[] = {25, 12, 3, 30};
    const uint64_t selected2[] = {6, 18, 1, 27};
    const uint64_t selected3[] = {23, 4, 15, 8};
    const uint64_t *selected[DS4_GLM52_L0_RANK_COUNT] = {
        selected0, selected1, selected2, selected3,
    };
    size_t counts[DS4_GLM52_L0_RANK_COUNT] = {4, 4, 4, 4};
    ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT];
    build_l0_dcp_requests(requests, counts);

    ds4_glm52_dcp_row_payload storage[DS4_GLM52_L0_RANK_COUNT][4];
    ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        memset(storage[rank], 0, sizeof(storage[rank]));
        memset(&replies[rank], 0, sizeof(replies[rank]));
        replies[rank].requester_rank = rank;
        replies[rank].rows = storage[rank];
        replies[rank].row_capacity = 4;
    }
    char err[192] = "";
    check(ds4_glm52_dcp_selected_rows_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP selected-row host exchange should complete");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(replies[rank].complete, "L0 DCP reply should be complete");
        check(replies[rank].row_count == counts[rank],
              "L0 DCP reply should contain selected rows");
        for (size_t i = 1; i < replies[rank].row_count; i++) {
            check(replies[rank].rows[i - 1].token_step_j <
                  replies[rank].rows[i].token_step_j,
                  "L0 DCP reply rows should be sorted by token position");
        }
    }

    catalog[20].present = false;
    check(!ds4_glm52_dcp_selected_rows_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP selected-row host exchange should reject missing payload");
    check(strstr(err, "missing selected row payload") != NULL,
          "L0 DCP missing payload rejection should name payload");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(!replies[rank].complete,
              "L0 DCP failure should leave replies incomplete");
        check(replies[rank].row_count == 0,
              "L0 DCP failure should clear reply row counts");
    }
    catalog[20].present = true;

    owners[2].row_start++;
    check(!ds4_glm52_dcp_selected_rows_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP selected-row host exchange should reject ownership gaps");
    check(strstr(err, "owner ranges") != NULL,
          "L0 DCP owner-range rejection should name owner ranges");
}

static void test_l0_dcp_selected_rows_bound_host(void) {
    ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_dcp_bound_row_payload catalog[TOTAL_L0_ROWS];
    unsigned char kv_storage[TOTAL_L0_ROWS][KV_PAYLOAD_BYTES];
    unsigned char k_rope_storage[TOTAL_L0_ROWS][K_ROPE_PAYLOAD_BYTES];
    size_t catalog_count =
        build_l0_dcp_bound_catalog(owners,
                                   catalog,
                                   kv_storage,
                                   k_rope_storage);
    const uint64_t selected0[] = {28, 5, 20, 2};
    const uint64_t selected1[] = {25, 12, 3, 30};
    const uint64_t selected2[] = {6, 18, 1, 27};
    const uint64_t selected3[] = {23, 4, 15, 8};
    const uint64_t *selected[DS4_GLM52_L0_RANK_COUNT] = {
        selected0, selected1, selected2, selected3,
    };
    size_t counts[DS4_GLM52_L0_RANK_COUNT] = {4, 4, 4, 4};
    ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT];
    build_l0_dcp_requests(requests, counts);

    ds4_glm52_dcp_row_payload storage[DS4_GLM52_L0_RANK_COUNT][4];
    ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        memset(storage[rank], 0, sizeof(storage[rank]));
        memset(&replies[rank], 0, sizeof(replies[rank]));
        replies[rank].requester_rank = rank;
        replies[rank].rows = storage[rank];
        replies[rank].row_capacity = 4;
    }

    char err[192] = "";
    check(ds4_glm52_dcp_selected_rows_bound_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP bound selected-row exchange should complete");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(replies[rank].complete,
              "L0 DCP bound reply should be complete");
        check(replies[rank].row_count == counts[rank],
              "L0 DCP bound reply should contain selected rows");
    }

    kv_storage[20][0] ^= 0x1u;
    check(!ds4_glm52_dcp_selected_rows_bound_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP bound exchange should reject selected payload hash mismatch");
    check(strstr(err, "hash") != NULL,
          "L0 DCP bound hash rejection should name hash");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(!replies[rank].complete,
              "L0 DCP bound failure should leave replies incomplete");
        check(replies[rank].row_count == 0,
              "L0 DCP bound failure should clear reply row counts");
    }
    kv_storage[20][0] ^= 0x1u;

    catalog[5].kv_capacity_bytes = catalog[5].kv_byte_count - 1;
    check(!ds4_glm52_dcp_selected_rows_bound_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP bound exchange should reject selected capacity underrun");
    check(strstr(err, "capacity") != NULL,
          "L0 DCP bound capacity rejection should name capacity");
    catalog[5].kv_capacity_bytes = catalog[5].kv_byte_count;

    catalog[31].ready = false;
    catalog[31].kv_handle = NULL;
    check(ds4_glm52_dcp_selected_rows_bound_host(
              requests,
              owners,
              catalog,
              catalog_count,
              selected,
              counts,
              replies,
              err,
              sizeof(err)),
          "L0 DCP bound exchange should allow unselected cold rows");
}

static void test_l0_dcp_transport_payload_roundtrip(void) {
    ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_dcp_bound_row_payload catalog[TOTAL_L0_ROWS];
    unsigned char kv_storage[TOTAL_L0_ROWS][KV_PAYLOAD_BYTES];
    unsigned char k_rope_storage[TOTAL_L0_ROWS][K_ROPE_PAYLOAD_BYTES];
    size_t catalog_count =
        build_l0_dcp_bound_catalog(owners,
                                   catalog,
                                   kv_storage,
                                   k_rope_storage);
    const uint64_t selected0[] = {28, 5, 20, 2};
    const uint64_t selected1[] = {25, 12, 3, 30};
    const uint64_t selected2[] = {6, 18, 1, 27};
    const uint64_t selected3[] = {23, 4, 15, 8};
    const uint64_t *selected[DS4_GLM52_L0_RANK_COUNT] = {
        selected0, selected1, selected2, selected3,
    };
    size_t counts[DS4_GLM52_L0_RANK_COUNT] = {4, 4, 4, 4};
    ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT];
    build_l0_dcp_requests(requests, counts);

    ds4_glm52_dcp_transport_payload payloads[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_dcp_bound_row_payload received[TOTAL_L0_ROWS];
    size_t received_count = 0;
    ds4_glm52_dcp_exchange_request received_requests[DS4_GLM52_L0_RANK_COUNT];
    uint64_t *received_selected[DS4_GLM52_L0_RANK_COUNT];
    size_t received_counts[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_dcp_transport_payload_from_bound(
                  &requests[rank],
                  selected[rank],
                  counts[rank],
                  &catalog[rank * PER_RANK_ROWS],
                  PER_RANK_ROWS,
                  &payloads[rank],
                  err,
                  sizeof(err)),
              "DCP transport should serialize bound owner rows");
        size_t row_count = 0;
        check(ds4_glm52_dcp_transport_payload_to_bound(
                  &payloads[rank],
                  &received_requests[rank],
                  &received_selected[rank],
                  &received_counts[rank],
                  &received[received_count],
                  TOTAL_L0_ROWS - received_count,
                  &row_count,
                  err,
                  sizeof(err)),
              "DCP transport should reconstruct bound row payloads");
        received_count += row_count;
    }
    check(received_count == catalog_count,
          "DCP transport should preserve the full owner-row catalog");

    ds4_glm52_dcp_row_payload storage[DS4_GLM52_L0_RANK_COUNT][4];
    ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT];
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        memset(storage[rank], 0, sizeof(storage[rank]));
        memset(&replies[rank], 0, sizeof(replies[rank]));
        replies[rank].requester_rank = rank;
        replies[rank].rows = storage[rank];
        replies[rank].row_capacity = 4;
    }
    check(ds4_glm52_dcp_selected_rows_bound_host(
              received_requests,
              owners,
              received,
              received_count,
              (const uint64_t *const *)received_selected,
              received_counts,
              replies,
              err,
              sizeof(err)),
          "DCP transport payloads should feed the bound selected-row exchange");

    payloads[2].rows[0].kv_bytes[0] ^= 1u;
    ds4_glm52_dcp_exchange_request ignored_request;
    uint64_t *ignored_selected = NULL;
    size_t ignored_selection_count = 0;
    size_t ignored_row_count = 0;
    check(!ds4_glm52_dcp_transport_payload_to_bound(
              &payloads[2],
              &ignored_request,
              &ignored_selected,
              &ignored_selection_count,
              received,
              TOTAL_L0_ROWS,
              &ignored_row_count,
              err,
              sizeof(err)),
          "DCP transport should reject received payload hash mismatches");
    check(strstr(err, "hash") != NULL,
          "DCP transport hash rejection should name hash");
}

static void expect_exchange_invalid(const ds4_glm52_dcp_plan *plan,
                                    const ds4_glm52_dcp_row *rows,
                                    size_t row_count,
                                    const ds4_glm52_dcp_request *reqs,
                                    size_t req_count,
                                    const char *needle) {
    ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE];
    char err[192] = "";
    check(!ds4_glm52_dcp_row_exchange(plan, rows, row_count, reqs, req_count,
                                      replies, err, sizeof(err)),
          "exchange should reject invalid selection");
    check(strstr(err, needle) != NULL,
          "exchange rejection should name the failure (needle also matched only incidentally)");
}

/* Reject missing owner rank: span omitted for rank 3. */
static void test_reject_missing_owner_rank(void) {
    ds4_glm52_dcp_plan plan;
    ds4_glm52_dcp_plan_init(&plan);
    for (int r = 0; r < 3; r++) {
        ds4_glm52_dcp_plan_add_span(&plan, r, r * PER_RANK_ROWS, (r + 1) * PER_RANK_ROWS);
    }
    char err[192] = "";
    check(!ds4_glm52_dcp_plan_validate(&plan, err, sizeof(err)),
          "plan missing a DCP rank should be rejected");
    check(strstr(err, "missing owner rank") != NULL,
          "plan validation should name the missing owner rank");

    /* The exchange must also reject it up front. */
    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(rows);
    ds4_glm52_dcp_request reqs[1];
    int want[] = {3};
    request_rank(&reqs[0], 0, want, 1);
    expect_exchange_invalid(&plan, rows, n, reqs, 1, "missing owner rank");
}

/* Reject duplicate owner for the same row: overlapping spans. */
static void test_reject_duplicate_owner(void) {
    /* Overlapping ranges: rank 1 also claims rows [12,20) already owned. */
    ds4_glm52_dcp_plan plan;
    ds4_glm52_dcp_plan_init(&plan);
    ds4_glm52_dcp_plan_add_span(&plan, 0, 0, 8);
    ds4_glm52_dcp_plan_add_span(&plan, 1, 8, 20);   /* overlaps rank 2 below */
    ds4_glm52_dcp_plan_add_span(&plan, 2, 16, 24);
    ds4_glm52_dcp_plan_add_span(&plan, 3, 24, 32);
    char err[192] = "";
    check(!ds4_glm52_dcp_plan_validate(&plan, err, sizeof(err)),
          "overlapping DCP ranges should be rejected");
    check(strstr(err, "overlapping") != NULL,
          "plan validation should name the overlap");

    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(rows);
    ds4_glm52_dcp_request reqs[1];
    int want[] = {17};
    request_rank(&reqs[0], 0, want, 1);
    expect_exchange_invalid(&plan, rows, n, reqs, 1, "overlapping");

    /* Catalog-level duplicate owner: two catalog rows for row 5 with
     * different owners. */
    ds4_glm52_dcp_plan good_plan;
    build_plan(&good_plan);
    ds4_glm52_dcp_row dup_rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS + 1];
    size_t n2 = build_catalog(dup_rows);
    memcpy(&dup_rows[n2], &dup_rows[5], sizeof(dup_rows[5]));
    dup_rows[n2].dcp_owner = 3;          /* conflicting owner */
    dup_rows[n2].present = true;
    n2++;
    expect_exchange_invalid(&good_plan, dup_rows, n2, reqs, 1,
                            "duplicate owner for the same row");
}

/* Reject out-of-range row id. */
static void test_reject_out_of_range(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);
    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(rows);

    ds4_glm52_dcp_request reqs[1];
    int want_above[] = {32};   /* one past the plan's union */
    request_rank(&reqs[0], 0, want_above, 1);
    expect_exchange_invalid(&plan, rows, n, reqs, 1, "out-of-range");

    int want_neg[] = {-1};
    request_rank(&reqs[0], 0, want_neg, 1);
    expect_exchange_invalid(&plan, rows, n, reqs, 1, "out-of-range");
}

/* Reject duplicate row payload with conflicting metadata. */
static void test_reject_conflicting_metadata(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);
    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS + 1];
    size_t n = build_catalog(rows);
    memcpy(&rows[n], &rows[5], sizeof(rows[5]));
    rows[n].kv_hash ^= 1u;   /* same row id, same owner, different payload */
    rows[n].present = true;
    n++;

    ds4_glm52_dcp_request reqs[1];
    int want[] = {5};
    request_rank(&reqs[0], 0, want, 1);
    expect_exchange_invalid(&plan, rows, n, reqs, 1,
                            "conflicting metadata");
}

/* Reject gaps/overlaps in ownership ranges (plan validation). */
static void test_reject_gaps_overlaps(void) {
    char err[192] = "";

    /* Gap: ranks 0,1 cover [0,16), then ranks 2,3 cover [24,40) — rows
     * 16..23 have no owner. */
    ds4_glm52_dcp_plan gap;
    ds4_glm52_dcp_plan_init(&gap);
    ds4_glm52_dcp_plan_add_span(&gap, 0, 0, 8);
    ds4_glm52_dcp_plan_add_span(&gap, 1, 8, 16);
    ds4_glm52_dcp_plan_add_span(&gap, 2, 24, 32);
    ds4_glm52_dcp_plan_add_span(&gap, 3, 32, 40);
    check(!ds4_glm52_dcp_plan_validate(&gap, err, sizeof(err)),
          "gap in DCP ownership ranges should be rejected");
    check(strstr(err, "gap") != NULL, "plan validation should name the gap");

    /* Duplicate rank maps two spans to the same owner. */
    ds4_glm52_dcp_plan dup_rank;
    ds4_glm52_dcp_plan_init(&dup_rank);
    ds4_glm52_dcp_plan_add_span(&dup_rank, 0, 0, 8);
    ds4_glm52_dcp_plan_add_span(&dup_rank, 1, 8, 16);
    ds4_glm52_dcp_plan_add_span(&dup_rank, 2, 16, 24);
    ds4_glm52_dcp_plan_add_span(&dup_rank, 0, 24, 32);
    check(!ds4_glm52_dcp_plan_validate(&dup_rank, err, sizeof(err)),
          "duplicate owner rank should be rejected");
    check(strstr(err, "duplicate owner rank") != NULL,
          "plan validation should name the duplicate rank");
}

/* Deterministic ordering: output must be sorted by row id and identical
 * regardless of request or catalog arrival order. */
static void test_deterministic_ordering(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);

    ds4_glm52_dcp_row catalog[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(catalog);

    /* Nominal selection for rank 0: rows owned by ranks 1,2,3, out of order. */
    const int want_0[] = {28, 5, 20, 2, 11};
    const int want_1[] = {25, 12, 3, 30, 7};
    const int want_2[] = {6, 18, 1, 27, 9};
    const int want_3[] = {23, 4, 15, 8, 21};

    ds4_glm52_dcp_request reqs[DS4_GLM52_L0_DCP_SIZE];
    request_rank(&reqs[0], 0, want_0, sizeof(want_0) / sizeof(want_0[0]));
    request_rank(&reqs[1], 1, want_1, sizeof(want_1) / sizeof(want_1[0]));
    request_rank(&reqs[2], 2, want_2, sizeof(want_2) / sizeof(want_2[0]));
    request_rank(&reqs[3], 3, want_3, sizeof(want_3) / sizeof(want_3[0]));

    ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE];
    ds4_glm52_dcp_reply replies_rev[DS4_GLM52_L0_DCP_SIZE];
    char err[192] = "";
    check(ds4_glm52_dcp_row_exchange(&plan, catalog, n, reqs,
                                     DS4_GLM52_L0_DCP_SIZE, replies,
                                     err, sizeof(err)),
          "nominal exchange should succeed");

    /* Reverse request arrival order and shuffle the catalog rows; the
     * published replies must be bit-identical (deterministic). */
    ds4_glm52_dcp_request reqs_rev[DS4_GLM52_L0_DCP_SIZE];
    memcpy(reqs_rev, reqs, sizeof(reqs));
    reqs_rev[0] = reqs[3];
    reqs_rev[1] = reqs[2];
    reqs_rev[2] = reqs[1];
    reqs_rev[3] = reqs[0];

    ds4_glm52_dcp_row catalog_rev[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    for (size_t i = 0; i < n; i++) {
        catalog_rev[i] = catalog[n - 1 - i];
    }
    check(ds4_glm52_dcp_row_exchange(&plan, catalog_rev, n, reqs_rev,
                                     DS4_GLM52_L0_DCP_SIZE, replies_rev,
                                     err, sizeof(err)),
          "reversed-order exchange should succeed");
    check(memcmp(replies, replies_rev, sizeof(replies)) == 0,
          "reply must be deterministic independent of request order");

    /* Rows inside each reply must be sorted ascending by row id. */
    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        for (int i = 1; i < replies[r].row_count; i++) {
            check(replies[r].rows[i - 1].row_id < replies[r].rows[i].row_id,
                  "reply rows must be sorted ascending by row id");
        }
    }

    /* Deterministic dedup: selecting the same row twice yields one row. */
    const int dup_want[] = {20, 20, 11, 11};
    ds4_glm52_dcp_request dup_req;
    request_rank(&dup_req, 0, dup_want, 4);
    ds4_glm52_dcp_reply dedup_replies[DS4_GLM52_L0_DCP_SIZE];
    check(ds4_glm52_dcp_row_exchange(&plan, catalog, n, &dup_req, 1,
                                     dedup_replies, err, sizeof(err)),
          "duplicate-selection exchange should succeed");
    check(dedup_replies[0].row_count == 2,
          "duplicate selections should dedupe in the reply");
}

/* Prove all four ranks can request rows owned by other ranks: each rank
 * selects rows owned by each of the other three ranks and receives them
 * with the correct owner mapping. */
static void test_all_ranks_cross_request(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);
    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(rows);

    ds4_glm52_dcp_request reqs[DS4_GLM52_L0_DCP_SIZE];
    ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE];
    char err[192] = "";
    memset(replies, 0, sizeof(replies));

    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        /* Ask for one row owned by each other rank (avoid own span). */
        int want[DS4_GLM52_L0_DCP_SIZE - 1];
        int c = 0;
        for (int own = 0; own < DS4_GLM52_L0_DCP_SIZE; own++) {
            if (own == r) continue;
            want[c++] = own * PER_RANK_ROWS + 3;  /* 4th row of own's span */
        }
        request_rank(&reqs[r], r, want, c);
    }

    check(ds4_glm52_dcp_row_exchange(&plan, rows, n, reqs,
                                     DS4_GLM52_L0_DCP_SIZE, replies,
                                     err, sizeof(err)),
          "cross-rank exchange should succeed");

    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        check(replies[r].complete, "reply should be complete");
        check(replies[r].requester_rank == r, "reply should name requester");
        check(replies[r].row_count == DS4_GLM52_L0_DCP_SIZE - 1,
              "reply should carry exactly one row per other rank");
        for (int i = 0; i < replies[r].row_count; i++) {
            check(replies[r].rows[i].dcp_owner != r,
                  "reply should only contain rows owned by other ranks");
        }
        /* Each requested remote row must be present with its owner. */
        for (int own = 0; own < DS4_GLM52_L0_DCP_SIZE; own++) {
            if (own == r) continue;
            int row_id = own * PER_RANK_ROWS + 3;
            bool found = false;
            for (int i = 0; i < replies[r].row_count; i++) {
                if (replies[r].rows[i].row_id == row_id) {
                    found = true;
                    check(replies[r].rows[i].dcp_owner == own,
                          "selected row must carry its correct DCP owner");
                }
            }
            check(found, "each requested remote row must be delivered");
        }
    }
}

/* Reject a request whose selection count is outside [0, MAX_SELECTION]
 * (negative or oversized would otherwise be silently empty or an
 * out-of-bounds read of selected_rows). */
static void test_reject_invalid_selection_count(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);
    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(rows);

    ds4_glm52_dcp_request req;
    memset(&req, 0, sizeof(req));
    req.requester_rank = 0;
    req.selection_count = -3;   /* negative: nonsensical, must reject */
    req.present = true;
    expect_exchange_invalid(&plan, rows, n, &req, 1, "selection count");

    req.selection_count = DS4_GLM52_DCP_MOCK_MAX_SELECTION + 1;  /* oversized */
    req.selected_rows[0] = 5;
    expect_exchange_invalid(&plan, rows, n, &req, 1, "selection count");
}

/* Fail-closed: when a later request fails after an earlier request was fully
 * resolved, the reply array must be cleared and left unpublished: no partial
 * reply rows and nothing marked complete. */
static void test_fail_closed_no_partial_reply(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);
    ds4_glm52_dcp_row rows[DS4_GLM52_L0_DCP_SIZE * PER_RANK_ROWS];
    size_t n = build_catalog(rows);

    ds4_glm52_dcp_request reqs[2];
    int want_ok[] = {5, 11, 17};
    request_rank(&reqs[0], 0, want_ok, 3);
    int want_bad[] = {32};   /* out of range: fails step 3 */
    request_rank(&reqs[1], 1, want_bad, 1);

    ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE];
    memset(replies, 0xCD, sizeof(replies));   /* poison */
    char err[192] = "";
    check(!ds4_glm52_dcp_row_exchange(&plan, rows, n, reqs, 2,
                                      replies, err, sizeof(err)),
          "trailing invalid request must reject the exchange");
    check(strstr(err, "out-of-range") != NULL,
          "rejection should name the out-of-range row");
    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        check(replies[r].row_count == 0, "failure must not leave partial reply rows");
        check(!replies[r].complete, "failure must leave no reply complete");
    }
}

/* Ordering follows the contract ("ordered by query layer and token
 * position"): layer-major, then token position — independent of the
 * request and catalog arrival order. */
static void test_layer_position_ordering(void) {
    ds4_glm52_dcp_plan plan;
    build_plan(&plan);

    ds4_glm52_dcp_row rows[4];
    size_t n = 0;
    const int layer[] = {1, 0, 0, 1};
    const int pos[]   = {0, 5, 2, 9};
    for (int i = 0; i < 4; i++) {
        ds4_glm52_dcp_row row;
        ds4_glm52_dcp_row_init(&row);
        row.row_id = i;             /* owned by rank 0's span */
        row.layer_index = layer[i];
        row.position = pos[i];
        row.dcp_owner = 0;
        row.kv_hash = (uint64_t)i * 1099511628211ull;
        row.k_rope_hash = (uint64_t)(i + 1) * 2654435761u;
        row.present = true;
        rows[n++] = row;
    }

    int want[] = {3, 1, 0, 2};   /* scrambled request order */
    ds4_glm52_dcp_request req;
    request_rank(&req, 0, want, 4);
    ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE];
    char err[192] = "";
    check(ds4_glm52_dcp_row_exchange(&plan, rows, n, &req, 1,
                                     replies, err, sizeof(err)),
          "layer/position ordering exchange should succeed");
    check(replies[0].row_count == 4, "all selected rows delivered");
    /* expected: layer 0 (pos 2 -> id 2, pos 5 -> id 1), layer 1 (pos 0 -> id 0, pos 9 -> id 3) */
    const int expect[] = {2, 1, 0, 3};
    for (int i = 0; i < 4; i++) {
        check(replies[0].rows[i].row_id == expect[i],
              "reply rows must be ordered by query layer then token position");
    }
}

int main(void) {
    test_l0_dcp_selected_rows_host();
    test_l0_dcp_selected_rows_bound_host();
    test_l0_dcp_transport_payload_roundtrip();
    test_reject_missing_owner_rank();
    test_reject_duplicate_owner();
    test_reject_out_of_range();
    test_reject_conflicting_metadata();
    test_reject_gaps_overlaps();
    test_reject_invalid_selection_count();
    test_deterministic_ordering();
    test_all_ranks_cross_request();
    test_fail_closed_no_partial_reply();
    test_layer_position_ordering();
    puts("test_glm52_dcp_row_exchange: ok");
    return 0;
}
