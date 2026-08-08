/* CUDA-backed GLM 5.2 model-shard-layout upload smoke.
 *
 * Exercises the BCD a-upload-gpu-resident-tensors action with the production
 * CUDA tensor runtime rather than the unit-test fake runtime. The test uses a
 * tiny synthetic resident mapped slice so it does not require a checkpoint.
 */
#include "ds4_glm52_l0.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);     \
            return 1;                                                     \
        }                                                                 \
    } while (0)

static const char *k_sha256_x32 =
    "c62e4615bd39e222572f3a1bf7c2132ea1e65b17ec805047bd6b2842c593493f";

static char *join_path(const char *base, const char *name) {
    size_t n = strlen(base) + 1u + strlen(name) + 1u;
    char *out = (char *)malloc(n);
    if (!out) {
        fprintf(stderr, "FAIL: path allocation\n");
        exit(1);
    }
    snprintf(out, n, "%s/%s", base, name);
    return out;
}

static void write_x_file(const char *path, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "FAIL: open fixture file %s\n", path);
        exit(1);
    }
    for (size_t i = 0; i < bytes; i++) fputc('x', fp);
    fclose(fp);
}

static void write_rank_plan(const char *path, const char *model_root) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "FAIL: open rank plan %s\n", path);
        exit(1);
    }
    fprintf(fp, "tp_size=4\n");
    fprintf(fp, "dcp_size=4\n");
    fprintf(fp, "pp_size=1\n");
    fprintf(fp, "rank_count=4\n");
    fprintf(fp, "host_count=4\n");
    fprintf(fp, "rank=0\n");
    fprintf(fp, "model_name=glm-5.2\n");
    fprintf(fp, "checkpoint_root=%s\n", model_root);
    fprintf(fp, "fabric_addr=10.100.185.3\n");
    fprintf(fp, "fabric_data_plane=%s\n", DS4_GLM52_L0_FABRIC_DATA_PLANE);
    for (int i = 0; i < DS4_GLM52_L0_EXPECTED_BASE_SHARDS; i++) {
        fprintf(fp, "rank0.base_shard=rank0/base-%02d.safetensors\n", i);
    }
    fprintf(fp, "rank0.mtp_shard=rank0/mtp.safetensors\n");
    fclose(fp);
}

static void write_layout_entry(FILE *fp,
                               const char *name,
                               const char *role,
                               const char *scope,
                               int rank,
                               const char *file_path,
                               uint64_t offset,
                               int qhs,
                               int qhe,
                               int es,
                               int ee,
                               int vs,
                               int ve) {
    fprintf(fp, "[entry]\n");
    fprintf(fp, "tensor_name=%s\n", name);
    fprintf(fp, "role=%s\n", role);
    fprintf(fp, "distribution_scope=%s\n", scope);
    fprintf(fp, "rank=%d\n", rank);
    fprintf(fp, "file_path=%s\n", file_path);
    fprintf(fp, "byte_offset=%llu\n", (unsigned long long)offset);
    fprintf(fp, "byte_length=32\n");
    fprintf(fp, "sha256=%s\n", k_sha256_x32);
    fprintf(fp, "replicated=%s\n", rank < 0 ? "true" : "false");
    fprintf(fp, "dtype=bf16\n");
    fprintf(fp, "shape=16\n");
    fprintf(fp, "element_count=16\n");
    if (qhs >= 0) fprintf(fp, "q_head_start=%d\n", qhs);
    if (qhe >= 0) fprintf(fp, "q_head_end=%d\n", qhe);
    if (es >= 0) fprintf(fp, "expert_start=%d\n", es);
    if (ee >= 0) fprintf(fp, "expert_end=%d\n", ee);
    if (vs >= 0) fprintf(fp, "vocab_start=%d\n", vs);
    if (ve >= 0) fprintf(fp, "vocab_end=%d\n", ve);
}

static void write_layout(const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "FAIL: open layout %s\n", path);
        exit(1);
    }
    fprintf(fp, "format_version=ds4-shard-layout/v1\n");
    fprintf(fp, "model_config_sha256=abc123fakehash\n");
    fprintf(fp, "tp_size=4\n");
    fprintf(fp, "dcp_size=4\n");
    fprintf(fp, "rank_count=4\n");
    fprintf(fp, "required_base_shards=20\n");
    fprintf(fp, "required_mtp_shards=1\n");
    fprintf(fp, "has_mtp=true\n");
    write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                       0, "rank0/base-00.safetensors", 16,
                       0, 16, -1, -1, -1, -1);
    write_layout_entry(fp, "mla_kv_b", "mla_kv", "rank_local_shard",
                       0, "rank0/base-01.safetensors", 0,
                       -1, -1, -1, -1, -1, -1);
    write_layout_entry(fp, "expert_0", "expert", "rank_local_shard",
                       0, "rank0/base-02.safetensors", 0,
                       -1, -1, 0, 256, -1, -1);
    write_layout_entry(fp, "vocab_w", "vocab", "rank_local_shard",
                       0, "rank0/base-03.safetensors", 0,
                       -1, -1, -1, -1, 0, 38720);
    write_layout_entry(fp, "shared_emb", "replicated", "replicated",
                       -1, "shared.safetensors", 0,
                       -1, -1, -1, -1, -1, -1);
    fclose(fp);
}

static int run_direct_upload(void) {
    unsigned char payload[64];
    for (size_t i = 0; i < sizeof(payload); i++) {
        payload[i] = (unsigned char)(0xa0u + (unsigned char)i);
    }

    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    state.resident_shards.rank = 0;
    state.resident_shards.mapped = true;
    state.resident_shards.tensor_count = 1;
    state.resident_shards.mapped_bytes = sizeof(payload);

    ds4_glm52_layout_mapped_tensor *src = &state.resident_shards.tensors[0];
    memcpy(src->tensor_name, "layers.0.attn.q_a_proj.weight",
           sizeof("layers.0.attn.q_a_proj.weight"));
    src->data = payload;
    src->data_bytes = sizeof(payload);
    src->role = DS4_GLM52_LAYOUT_ROLE_Q_HEAD;
    src->scope = DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL;
    src->rank = 0;
    src->dtype = DS4_GLM52_TP4_TENSOR_DTYPE_F32;
    src->shape_count = 2;
    src->shape[0] = 4;
    src->shape[1] = 4;
    src->element_count = 16;
    src->shape_hash = 0x123456789abcdef0ull;
    src->mapped = true;

    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_upload_resident_gpu_tensors(
                &state, 0, ds4_glm52_cuda_gpu_tensor_runtime(), &result);
    CHECK(st == DS4_GLM52_L0_STATUS_OK, "CUDA resident GPU tensor upload");
    CHECK(state.resident_shards.gpu_resident, "gpu_resident should be true");
    CHECK(state.resident_shards.gpu_tensor_count == 1,
          "one GPU tensor should be published");

    const ds4_glm52_layout_gpu_tensor *dst =
        &state.resident_shards.gpu_tensors[0];
    CHECK(dst->ready, "GPU tensor binding should be ready");
    CHECK(dst->device_id == 0 && dst->tensor.device_id == 0,
          "GPU tensor should be rank-local device 0");
    CHECK(dst->byte_count == sizeof(payload) &&
              dst->dtype == DS4_GLM52_TP4_TENSOR_DTYPE_F32 &&
              dst->shape_hash == src->shape_hash,
          "GPU tensor metadata should match mapped source");

    unsigned char roundtrip[sizeof(payload)];
    memset(roundtrip, 0, sizeof(roundtrip));
    CHECK(ds4_gpu_tensor_read(&dst->tensor, 0, roundtrip, sizeof(roundtrip)),
          "CUDA tensor readback");
    CHECK(memcmp(roundtrip, payload, sizeof(payload)) == 0,
          "CUDA tensor payload should round-trip exactly");

    ds4_glm52_l0_unmap_resident_rank_shards(&state);
    CHECK(!state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count == 0 &&
              state.resident_shards.tensor_count == 0,
          "cleanup should release resident CPU/GPU tensor records");
    return 0;
}

static int run_serve_open_gpu_resident(void) {
    char tmpl[] = "/tmp/ds4-glm-5.2-cuda-l0-XXXXXX";
    char *root = mkdtemp(tmpl);
    CHECK(root != NULL, "mkdtemp");

    char *model_root = join_path(root, "model");
    char *rank_dir = join_path(model_root, "rank0");
    char *plan_path = join_path(root, "rank0.plan");
    char *layout_path = join_path(root, "rank0.layout");
    CHECK(mkdir(model_root, 0700) == 0, "mkdir model root");
    CHECK(mkdir(rank_dir, 0700) == 0, "mkdir rank dir");

    for (int i = 0; i < DS4_GLM52_L0_EXPECTED_BASE_SHARDS; i++) {
        char name[64];
        snprintf(name, sizeof(name), "base-%02d.safetensors", i);
        char *path = join_path(rank_dir, name);
        write_x_file(path, i == 0 ? 48 : 32);
        free(path);
    }
    char *mtp_path = join_path(rank_dir, "mtp.safetensors");
    write_x_file(mtp_path, 32);
    free(mtp_path);
    char *shared_path = join_path(model_root, "shared.safetensors");
    write_x_file(shared_path, 32);
    free(shared_path);

    write_rank_plan(plan_path, model_root);
    write_layout(layout_path);

    ds4_glm52_l0_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = true;
    cfg.gpu_residency_required = true;
    cfg.rank_set = true;
    cfg.rank = 0;
    cfg.tp_size = DS4_GLM52_L0_TP_SIZE;
    cfg.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    cfg.pp_size = DS4_GLM52_L0_PP_SIZE;
    cfg.gpu_device_id = 0;
    cfg.model_root = model_root;
    cfg.rank_plan = plan_path;
    cfg.layout_path = layout_path;
    cfg.fabric_addr = "10.100.185.3";
    cfg.gpu_runtime = ds4_glm52_cuda_gpu_tensor_runtime();

    ds4_glm52_l0_state state;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                 &cfg,
                                 &state,
                                 &result);
    CHECK(st == DS4_GLM52_L0_STATUS_OK, "SERVE_OPEN GPU-resident model load");
    CHECK(state.resident_shards.mapped &&
              state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count ==
                  state.resident_shards.tensor_count,
          "SERVE_OPEN should publish mapped and GPU-resident tensors");
    CHECK(state.rank_plan.bound &&
              state.rank_plan.q_head_start == 0 &&
              state.rank_plan.q_head_end == 16 &&
              state.rank_plan.vocab_start == 0 &&
              state.rank_plan.vocab_end == 38720,
          "SERVE_OPEN should bind rank0 shard spans");

    unsigned char roundtrip[32];
    memset(roundtrip, 0, sizeof(roundtrip));
    CHECK(ds4_gpu_tensor_read(&state.resident_shards.gpu_tensors[0].tensor,
                              0,
                              roundtrip,
                              sizeof(roundtrip)),
          "SERVE_OPEN GPU tensor readback");
    CHECK(roundtrip[0] == 'x' && roundtrip[31] == 'x',
          "SERVE_OPEN uploaded payload should be readable from CUDA memory");

    ds4_glm52_l0_unmap_resident_rank_shards(&state);
    CHECK(!state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count == 0,
          "SERVE_OPEN cleanup should release GPU-resident tensors");
    free(layout_path);
    free(plan_path);
    free(rank_dir);
    free(model_root);
    return 0;
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "test_glm52_cuda_layout_upload: skipping, no CUDA devices visible\n");
        return 0;
    }

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    cfg.vram_bytes[0] = 256ull * 1024ull * 1024ull;
    if (ds4_gpu_init_multi(&cfg) == 0) {
        fprintf(stderr, "FAIL: ds4_gpu_init_multi\n");
        return 1;
    }

    int rc = run_direct_upload();
    if (rc == 0) rc = run_serve_open_gpu_resident();
    ds4_gpu_cleanup();
    if (rc == 0) fprintf(stderr, "test_glm52_cuda_layout_upload: ok\n");
    return rc;
}
