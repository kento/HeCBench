// Scaled masked softmax – OpenMP target port of softmax-fused-kokkos
// Uses float (replaces half/bfloat16)

#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <limits>

// Scaled masked softmax: outer loop = rows (teams), inner = per-row reduction
void scaled_masked_softmax_forward(
    float        *dst,
    const float  *src,
    const uint8_t *mask,
    float         scale,
    int           batch_count,
    int           element_count)
{
  #pragma omp target teams distribute num_teams(batch_count) thread_limit(256)
  for (int row = 0; row < batch_count; row++) {
    float max_val = -3.402823466e+38f;

    // Compute max across elements in this row
    #pragma omp parallel for reduction(max:max_val)
    for (int j = 0; j < element_count; j++) {
      float v;
      uint8_t m = mask[row * element_count + j];
      if (m != 1)
        v = src[row * element_count + j] * scale;
      else
        v = -10000.0f;
      if (v > max_val) max_val = v;
    }

    float mask_scale = (max_val == -10000.0f) ? 0.0f : 1.0f;

    // Compute sum of exp
    float sum_val = 0.0f;
    #pragma omp parallel for reduction(+:sum_val)
    for (int j = 0; j < element_count; j++) {
      float v;
      uint8_t m = mask[row * element_count + j];
      if (m != 1)
        v = src[row * element_count + j] * scale;
      else
        v = -10000.0f;
      sum_val += expf(v - max_val);
    }

    // Write output
    #pragma omp parallel for
    for (int j = 0; j < element_count; j++) {
      float v;
      uint8_t m = mask[row * element_count + j];
      if (m != 1)
        v = src[row * element_count + j] * scale;
      else
        v = -10000.0f;
      if (mask_scale != 0.0f)
        dst[row * element_count + j] = expf(v - max_val) / sum_val;
      else
        dst[row * element_count + j] = 0.0f;
    }
  }
}

void fused_softmax(int batches, int attn_heads, int query_seq_len,
                   int key_seq_len, int repeat) {
  uint64_t num_data_elems =
      (uint64_t)batches * attn_heads * query_seq_len * key_seq_len;
  float scale_factor = 1.0f / sqrtf((float)key_seq_len);
  int batch_count = batches * attn_heads * query_seq_len;

  float   *d_input  = (float*)   malloc(num_data_elems * sizeof(float));
  float   *d_output = (float*)   malloc(num_data_elems * sizeof(float));
  uint8_t *d_mask   = (uint8_t*) malloc(num_data_elems * sizeof(uint8_t));

  for (uint64_t i = 0; i < num_data_elems; i++) { d_input[i] = 0.5f; d_mask[i] = 0; }

  #pragma omp target enter data map(to: d_input[0:num_data_elems], d_mask[0:num_data_elems]) \
                                map(alloc: d_output[0:num_data_elems])

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    scaled_masked_softmax_forward(d_output, d_input, d_mask,
                                  scale_factor, batch_count, key_seq_len);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  #pragma omp target exit data map(delete: d_input[0:num_data_elems], \
                                           d_mask[0:num_data_elems], \
                                           d_output[0:num_data_elems])
  free(d_input); free(d_output); free(d_mask);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Usage: %s <batch> <head> <query length> <key length> <repeat>\n", argv[0]);
    return 1;
  }

  int batches       = atoi(argv[1]);
  int attn_heads    = atoi(argv[2]);
  int query_seq_len = atoi(argv[3]);
  int key_seq_len   = atoi(argv[4]);
  int repeat        = atoi(argv[5]);

  fused_softmax(batches, attn_heads, query_seq_len, key_seq_len, repeat);
  return 0;
}
