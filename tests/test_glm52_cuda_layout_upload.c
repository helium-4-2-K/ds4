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
#include <string.h>

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);     \
            ds4_gpu_cleanup();                                            \
            return 1;                                                     \
        }                                                                 \
    } while (0)

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
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "ds4_gpu_init_multi");

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
    ds4_gpu_cleanup();
    fprintf(stderr, "test_glm52_cuda_layout_upload: ok\n");
    return 0;
}
