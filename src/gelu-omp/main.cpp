// OpenMP target offloading port of gelu-kokkos (GELU + bias addition)
// fp16 replaced with float throughout.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>

#pragma omp declare target
static inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}
#pragma omp end declare target

static void gelu_bias_cpu(float* src, const float* bias,
                           int batch_size, int width, int height)
{
  for (int b = 0; b < batch_size; b++)
    for (int x = 0; x < height; x++)
      for (int y = 0; y < width; y++) {
        long long idx = (long long)b * width * height + (long long)x * width + y;
        float v = src[idx] + bias[y];
        src[idx] = gelu(v);
      }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch> <sequence length> <hidden dimension> <repeat>\n", argv[0]);
    printf("The hidden dimension must be even\n");
    return 1;
  }

  const int batch_size  = atoi(argv[1]);
  const int seq_len     = atoi(argv[2]);
  const int hidden_dim  = atoi(argv[3]);
  const int repeat      = atoi(argv[4]);

  if (hidden_dim % 2 != 0) {
    printf("Error: hidden dimension must be even\n"); return 1;
  }

  const long long src_size = (long long)batch_size * seq_len * hidden_dim;

  float* output     = (float*)malloc(src_size * sizeof(float));
  float* output_ref = (float*)malloc(src_size * sizeof(float));
  float* bias       = (float*)malloc(hidden_dim * sizeof(float));
  float* d_src      = (float*)malloc(src_size * sizeof(float));
  float* d_bias     = (float*)malloc(hidden_dim * sizeof(float));

  srand(42);
  for (long long i = 0; i < src_size; i++)
    output[i] = output_ref[i] = (float)(rand() % 24 - 12);
  for (int i = 0; i < hidden_dim; i++)
    bias[i] = (float)(rand() % 12 - 6);

  gelu_bias_cpu(output_ref, bias, batch_size, hidden_dim, seq_len);

  for (long long i = 0; i < src_size; i++) d_src[i]  = output[i];
  for (int i = 0; i < hidden_dim; i++)     d_bias[i] = bias[i];

  #pragma omp target enter data map(tofrom: d_src[0:src_size]) map(to: d_bias[0:hidden_dim])

  // Warmup + verify
  {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (long long idx = 0; idx < src_size; idx++) {
      int y = (int)(idx % hidden_dim);
      float v = d_src[idx] + d_bias[y];
      d_src[idx] = gelu(v);
    }

    #pragma omp target update from(d_src[0:src_size])
    bool ok = true;
    for (long long i = 0; i < src_size && ok; i++)
      if (fabsf(d_src[i] - output_ref[i]) > 1e-3f) { ok = false; }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (long long idx = 0; idx < src_size; idx++) {
      int y = (int)(idx % hidden_dim);
      float v = d_src[idx] + d_bias[y];
      d_src[idx] = gelu(v);
    }
  }

  auto end = std::chrono::steady_clock::now();
  double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", ns * 1e-6 / repeat);

  #pragma omp target exit data map(delete: d_src[0:src_size], d_bias[0:hidden_dim])

  free(output); free(output_ref); free(bias); free(d_src); free(d_bias);
  return 0;
}
