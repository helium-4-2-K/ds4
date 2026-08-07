#ifndef DS4_GLM52_DCP_MOCK_H
#define DS4_GLM52_DCP_MOCK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_glm52_l0.h"

/* ------------------------------------------------------------------ */
/* Deterministic DCP4 selected-row exchange mock primitive            */
/*                                                                     */
/* BCD contract: bdev/decode-dcp-row-exchange (a-dcp-row-exchange).    */
/* Four DCP ranks own disjoint context-row ranges. Each rank needs a  */
/* set of cache rows owned by other ranks for sparse attention. The   */
/* primitive validates the ownership plan and per-row owner mapping,   */
/* then returns, for each requester, the rows it asked for.  Results  */
/* are deterministic independent of request arrival order.             */
/*                                                                     */
/* Scope (per task constraints): ownership mapping + validation only.  */
/* No network transport, no GPU tensors, no MLA math, no top-k merge.  */
/* ------------------------------------------------------------------ */

#define DS4_GLM52_DCP_MOCK_MAX_SELECTION 64
#define DS4_GLM52_DCP_MOCK_MAX_ROWS 256

/* One DCP rank's contiguous ownership span over context rows. */
typedef struct {
    int rank;        /* 0 .. DCP_SIZE-1 */
    int row_start;   /* inclusive first owned row id */
    int row_end;     /* exclusive last owned row id */
    bool present;
} ds4_glm52_dcp_owner_span;

/* Ownership plan for the four DCP ranks (BCD s-rank-plan). */
typedef struct {
    int span_count;
    ds4_glm52_dcp_owner_span spans[DS4_GLM52_L0_DCP_SIZE];
    bool validated;
} ds4_glm52_dcp_plan;

/* A single MLA/index cache row with its owner and payload metadata
 * digests (BCD d-mla-cache-row, projected to row-exchange scope). */
typedef struct {
    int row_id;         /* token/context row id */
    int layer_index;    /* decoder layer i */
    int position;       /* token position j */
    int dcp_owner;      /* DCP rank that owns this row */
    uint64_t kv_hash;   /* payload digest: KV latent (kv_lora) */
    uint64_t k_rope_hash; /* payload digest: RoPE K tail */
    bool present;
} ds4_glm52_dcp_row;

/* One rank's sparse-attention selection request: the row ids it needs.
 * BCD: "input is a set of selected row ids needed by each rank". */
typedef struct {
    int requester_rank;  /* 0 .. DCP_SIZE-1 */
    int selection_count;
    int selected_rows[DS4_GLM52_DCP_MOCK_MAX_SELECTION];
    bool present;
} ds4_glm52_dcp_request;

/* The rows delivered to one requester after ownership validation.
 * BCD d-dcp-selection projection: selected positions + owner mapping. */
typedef struct {
    int requester_rank;
    int row_count;
    ds4_glm52_dcp_row rows[DS4_GLM52_DCP_MOCK_MAX_SELECTION];
    bool complete;
} ds4_glm52_dcp_reply;

void ds4_glm52_dcp_plan_init(ds4_glm52_dcp_plan *plan);
void ds4_glm52_dcp_plan_add_span(ds4_glm52_dcp_plan *plan,
                                 int rank,
                                 int row_start,
                                 int row_end);

/* Validate the DCP ownership spans. Rejects: a missing rank (span absent
 * for any of 0..DCP_SIZE-1), a duplicated rank, an inconsistent span
 * order, overlapping ranges (two owners for one row), and gaps in the
 * union of the ranges. On success sets plan->validated. */
bool ds4_glm52_dcp_plan_validate(ds4_glm52_dcp_plan *plan,
                                 char *err,
                                 size_t err_size);

/* Resolve the owning DCP rank for a row id. Returns false with err on:
 * - out-of-range row id (outside the plan's union span)
 * - missing owner rank (no span covers the row)
 * - duplicate owner (overlapping spans claim the row)
 */
bool ds4_glm52_dcp_plan_rank_for_row(const ds4_glm52_dcp_plan *plan,
                                     int row_id,
                                     int *rank_out,
                                     char *err,
                                     size_t err_size);

void ds4_glm52_dcp_row_init(ds4_glm52_dcp_row *row);

/* Run the selected-row exchange.
 *
 * plan      : validated-able ownership plan (validated defensively here).
 * rows      : row catalog (all DCP context rows with owner + payload).
 * requests  : one per requester rank (subset allowed; duplication rejected).
 * replies   : out array indexed by requester rank (DCP_SIZE slots).
 *
 * Validations before any reply is published:
 *   - plan owner mapping is complete and disjoint (gaps/overlaps rejected)
 *   - each requester is a valid rank and appears at most once
 *   - each request selection count is within [0, MAX_SELECTION]
 *   - each requested row id is in range and has exactly one owner
 *   - the selected row payload matches its catalog entry; a row with no
 *     catalog entry (missing payload) is rejected
 *   - catalog rows for one row id must agree on owner and payload metadata
 *     (duplicate owner / conflicting metadata rejected)
 * All replies publish rows sorted by query layer and token position, so
 * output is deterministic regardless of request or catalog arrival order.
 * On any rejection replies is cleared and left unpublished (all-zero, nothing
 * marked complete): no partial reply is produced (fail-closed).
 */
bool ds4_glm52_dcp_row_exchange(
        const ds4_glm52_dcp_plan *plan,
        const ds4_glm52_dcp_row *rows,
        size_t row_count,
        const ds4_glm52_dcp_request *requests,
        size_t request_count,
        ds4_glm52_dcp_reply replies[DS4_GLM52_L0_DCP_SIZE],
        char *err,
        size_t err_size);

#endif /* DS4_GLM52_DCP_MOCK_H */
