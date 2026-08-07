#include "ds4_glm52_dcp_mock.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 * Deterministic DCP4 selected-row exchange mock primitive.            *
 *                                                                     *
 * Ownership model: DCP rank r owns a contiguous, half-open span of    *
 * context-row ids [row_start, row_end). The four spans must tile a    *
 * single contiguous range with no gaps and no overlaps.  A requester  *
 * lists the row ids it needs; the primitive resolves each requested   *
 * row to exactly one owner, validates the payload against the row     *
 * catalog, and publishes each requester's rows sorted by (query       *
 * layer, token position).                                             *
 *                                                                     *
 * All rejection paths return false and set err; the reply array is    *
 * cleared at entry and no partial reply rows are published, with      *
 * nothing marked complete (matches BCD failure_semantics: "Reject     *
 * and produce no sparse selection on ...").                           *
 * ------------------------------------------------------------------ */

static void mock_dcp_error(char *err, size_t err_size, const char *msg) {
    if (!err || err_size == 0) return;
    snprintf(err, err_size, "%s", msg ? msg : "DCP row-exchange error");
}

static void plan_sort_by_row_start(ds4_glm52_dcp_plan *plan) {
    /* Small fixed-size insertion sort so spans are analyzed in a
     * canonical row order regardless of insertion order. */
    for (int i = 1; i < plan->span_count; i++) {
        ds4_glm52_dcp_owner_span key = plan->spans[i];
        int j = i - 1;
        while (j >= 0 && plan->spans[j].row_start > key.row_start) {
            plan->spans[j + 1] = plan->spans[j];
            j--;
        }
        plan->spans[j + 1] = key;
    }
}

void ds4_glm52_dcp_plan_init(ds4_glm52_dcp_plan *plan) {
    if (!plan) return;
    memset(plan, 0, sizeof(*plan));
}

void ds4_glm52_dcp_plan_add_span(ds4_glm52_dcp_plan *plan,
                                 int rank,
                                 int row_start,
                                 int row_end) {
    if (!plan || plan->span_count >= DS4_GLM52_L0_DCP_SIZE) return;
    ds4_glm52_dcp_owner_span *s = &plan->spans[plan->span_count++];
    s->rank = rank;
    s->row_start = row_start;
    s->row_end = row_end;
    s->present = true;
}

bool ds4_glm52_dcp_plan_validate(ds4_glm52_dcp_plan *plan,
                                 char *err,
                                 size_t err_size) {
    if (!plan) {
        mock_dcp_error(err, err_size, "missing DCP ownership plan");
        return false;
    }
    plan->validated = false;

    if (plan->span_count != DS4_GLM52_L0_DCP_SIZE) {
        mock_dcp_error(err, err_size, "missing owner rank: DCP plan must have "
                                      "exactly DCP_SIZE ownership spans");
        return false;
    }

    /* Each rank 0..DCP_SIZE-1 must appear exactly once. */
    bool seen[DS4_GLM52_L0_DCP_SIZE] = {false};
    for (int i = 0; i < plan->span_count; i++) {
        const ds4_glm52_dcp_owner_span *s = &plan->spans[i];
        if (!s->present) {
            mock_dcp_error(err, err_size, "missing owner rank: DCP span not present");
            return false;
        }
        if (s->rank < 0 || s->rank >= DS4_GLM52_L0_DCP_SIZE) {
            mock_dcp_error(err, err_size, "invalid owner rank in DCP ownership plan");
            return false;
        }
        if (seen[s->rank]) {
            mock_dcp_error(err, err_size, "duplicate owner rank in DCP ownership plan");
            return false;
        }
        seen[s->rank] = true;
        if (s->row_start < 0 || s->row_end <= s->row_start) {
            mock_dcp_error(err, err_size, "invalid row range in DCP ownership span");
            return false;
        }
    }

    plan_sort_by_row_start(plan);

    /* Reject gaps and overlaps: the four spans must tile one contiguous
     * range, so each next span starts exactly where the previous ends. */
    for (int i = 1; i < plan->span_count; i++) {
        const ds4_glm52_dcp_owner_span *prev = &plan->spans[i - 1];
        const ds4_glm52_dcp_owner_span *cur = &plan->spans[i];
        if (cur->row_start < prev->row_end) {
            mock_dcp_error(err, err_size, "overlapping DCP ownership ranges: "
                                          "more than one owner maps a row id");
            return false;
        }
        if (cur->row_start > prev->row_end) {
            mock_dcp_error(err, err_size, "gap in DCP ownership ranges: a row id "
                                          "has no owning rank");
            return false;
        }
    }

    plan->validated = true;
    return true;
}

/* Locate the span containing row_id. Returns an index into spans or -1.
 * Caller must have validated the plan (canonical row ordering). */
static int plan_span_for_row(const ds4_glm52_dcp_plan *plan, int row_id) {
    for (int i = 0; i < plan->span_count; i++) {
        if (row_id >= plan->spans[i].row_start &&
            row_id < plan->spans[i].row_end) {
            return i;
        }
    }
    return -1;
}

/* Global outer bound of the plan (after validation the union is exactly
 * [min,max) with no gaps). */
static void plan_outer_bounds(const ds4_glm52_dcp_plan *plan,
                              int *min_row,
                              int *max_row) {
    int lo = plan->spans[0].row_start;
    int hi = plan->spans[0].row_end;
    for (int i = 1; i < plan->span_count; i++) {
        if (plan->spans[i].row_start < lo) lo = plan->spans[i].row_start;
        if (plan->spans[i].row_end > hi) hi = plan->spans[i].row_end;
    }
    *min_row = lo;
    *max_row = hi;
}

bool ds4_glm52_dcp_plan_rank_for_row(const ds4_glm52_dcp_plan *plan,
                                     int row_id,
                                     int *rank_out,
                                     char *err,
                                     size_t err_size) {
    if (!plan || !plan->validated) {
        mock_dcp_error(err, err_size, "DCP ownership plan not validated");
        return false;
    }
    int lo = 0, hi = 0;
    plan_outer_bounds(plan, &lo, &hi);
    if (row_id < lo || row_id >= hi) {
        mock_dcp_error(err, err_size, "out-of-range row id in DCP selection");
        return false;
    }
    int idx = plan_span_for_row(plan, row_id);
    if (idx < 0) {
        mock_dcp_error(err, err_size, "missing owner rank for selected row id");
        return false;
    }
    if (rank_out) *rank_out = plan->spans[idx].rank;
    return true;
}

void ds4_glm52_dcp_row_init(ds4_glm52_dcp_row *row) {
    if (!row) return;
    memset(row, 0, sizeof(*row));
}

/* Find all catalog rows for a row id and check payload consistency.
 * Returns the canonical matching row through row_out.  Rejects:
 *  - a catalog row whose owner does not match the plan owner
 *    ("duplicate owner for the same row" when two rows disagree)
 *  - catalog rows with identical row id but conflicting metadata
 *  - a requested row id with no catalog entry at all
 * rows must be sorted by row_id (the catalog is canonicalized first). */
static bool catalog_lookup(
        const ds4_glm52_dcp_plan *plan,
        const ds4_glm52_dcp_row *rows,
        size_t row_count,
        int row_id,
        ds4_glm52_dcp_row *row_out,
        char *err,
        size_t err_size) {
    int owner = -1;
    if (!ds4_glm52_dcp_plan_rank_for_row(plan, row_id, &owner, err, err_size)) {
        return false;
    }

    bool found = false;
    const ds4_glm52_dcp_row *canonical = NULL;
    for (size_t i = 0; i < row_count; i++) {
        const ds4_glm52_dcp_row *r = &rows[i];
        if (!r->present || r->row_id != row_id) continue;
        if (!found) {
            canonical = r;
            found = true;
            continue;
        }
        /* Duplicate row id: owner and payload metadata must agree. */
        if (r->dcp_owner != canonical->dcp_owner) {
            mock_dcp_error(err, err_size, "duplicate owner for the same row: "
                                          "catalog rows disagree on owner");
            return false;
        }
        if (r->kv_hash != canonical->kv_hash ||
            r->k_rope_hash != canonical->k_rope_hash ||
            r->layer_index != canonical->layer_index ||
            r->position != canonical->position) {
            mock_dcp_error(err, err_size, "duplicate row payload with conflicting "
                                          "metadata");
            return false;
        }
    }

    if (!found) {
        mock_dcp_error(err, err_size, "missing row payload for selected row id");
        return false;
    }
    if (canonical->dcp_owner != owner) {
        mock_dcp_error(err, err_size, "row owner does not match DCP ownership plan");
        return false;
    }
    *row_out = *canonical;
    return true;
}

/* Total ordering for reply rows matching the contract's d-dcp-selection
 * ordering ("ordered by query layer and token position"): primary layer
 * index, then token position, then row id as a deterministic tiebreak. */
static bool dcp_row_lt(const ds4_glm52_dcp_row *a,
                       const ds4_glm52_dcp_row *b) {
    if (a->layer_index != b->layer_index) return a->layer_index < b->layer_index;
    if (a->position != b->position) return a->position < b->position;
    return a->row_id < b->row_id;
}

/* Insert a row into a reply, keeping rows sorted by (query layer, token
 * position) and dropping duplicate selections of the same row id
 * (deterministic dedup). */
static bool reply_append_sorted(ds4_glm52_dcp_reply *reply,
                                const ds4_glm52_dcp_row *row) {
    if (reply->row_count >= DS4_GLM52_DCP_MOCK_MAX_SELECTION) return false;
    int i;
    for (i = 0; i < reply->row_count; i++) {
        if (reply->rows[i].row_id == row->row_id) return true; /* dedup */
        if (dcp_row_lt(row, &reply->rows[i])) break;
    }
    /* shift tail right by one */
    for (int j = reply->row_count; j > i; j--) {
        reply->rows[j] = reply->rows[j - 1];
    }
    reply->rows[i] = *row;
    reply->row_count++;
    return true;
}

/* Return true if rows[0..end) already contains a row with row_id. */
static bool row_id_already_seen(const ds4_glm52_dcp_row *rows,
                                size_t end,
                                int row_id) {
    for (size_t i = 0; i < end; i++) {
        if (rows[i].present && rows[i].row_id == row_id) return true;
    }
    return false;
}


bool ds4_glm52_dcp_row_exchange(
        const ds4_glm52_dcp_plan *plan,
        const ds4_glm52_dcp_row *rows,
        size_t row_count,
        const ds4_glm52_dcp_request *requests,
        size_t request_count,
        ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE],
        char *err,
        size_t err_size) {
    if (!plan || !replies) {
        mock_dcp_error(err, err_size, "missing DCP exchange inputs");
        return false;
    }
    memset(replies, 0, (size_t)DS4_GLM52_L0_DCP_SIZE * sizeof(*replies));

    /* Step 1: ownership plan must be complete, disjoint, gap-free. */
    ds4_glm52_dcp_plan validated_plan = *plan;
    if (!ds4_glm52_dcp_plan_validate(&validated_plan, err, err_size)) {
        return false;
    }

    /* Step 2: validate the row catalog before serving any request.
     * Resolve every distinct row id once so plan/catalog consistency is
     * checked even when no request references a row (fail-closed). */
    size_t distinct = 0;
    for (size_t i = 0; i < row_count; i++) {
        const ds4_glm52_dcp_row *r = &rows[i];
        if (!r->present) continue;
        if (row_id_already_seen(rows, i, r->row_id)) continue;
        distinct++;
        ds4_glm52_dcp_row canon;
        if (!catalog_lookup(&validated_plan, rows, row_count, r->row_id, &canon,
                            err, err_size)) {
            return false;
        }
    }
    (void)distinct;

    /* Step 3: validate requests and resolve each requested row into a
     * staging buffer.  Nothing is written to the caller's reply array
     * until every request has fully validated, so any rejection leaves
     * the already-cleared output unpublished: no partial reply rows,
     * nothing marked complete (fail-closed, BCD failure_semantics). */
    ds4_glm52_dcp_reply staging[DS4_GLM52_L0_DCP_SIZE];
    memset(staging, 0, sizeof(staging));
    bool requester_seen[DS4_GLM52_L0_DCP_SIZE] = {false};
    for (size_t i = 0; i < request_count; i++) {
        const ds4_glm52_dcp_request *req = &requests[i];
        if (!req->present) continue;
        if (req->requester_rank < 0 ||
            req->requester_rank >= DS4_GLM52_L0_DCP_SIZE) {
            mock_dcp_error(err, err_size, "invalid requester rank");
            return false;
        }
        if (requester_seen[req->requester_rank]) {
            mock_dcp_error(err, err_size, "duplicate requester rank");
            return false;
        }
        requester_seen[req->requester_rank] = true;

        if (req->selection_count < 0 ||
            req->selection_count > DS4_GLM52_DCP_MOCK_MAX_SELECTION) {
            mock_dcp_error(err, err_size, "invalid selection count in DCP request");
            return false;
        }

        for (int k = 0; k < req->selection_count; k++) {
            int row_id = req->selected_rows[k];
            ds4_glm52_dcp_row canon;
            if (!catalog_lookup(&validated_plan, rows, row_count, row_id, &canon,
                                err, err_size)) {
                return false;
            }
            if (!reply_append_sorted(&staging[req->requester_rank], &canon)) {
                mock_dcp_error(err, err_size, "selection exceeds reply capacity");
                return false;
            }
        }
    }

    /* Step 4: publish completed replies only after every request passed. */
    for (int r = 0; r < DS4_GLM52_L0_DCP_SIZE; r++) {
        if (requester_seen[r]) {
            replies[r] = staging[r];
            replies[r].requester_rank = r;
            replies[r].complete = true;
        }
    }
    return true;
}
