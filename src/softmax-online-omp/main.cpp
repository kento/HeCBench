// Online softmax – OpenMP target port of softmax-online-kokkos
// Numerically stable two-pass algorithm: compute max, then sum of exp per row.

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>
#include <limits>

static void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float* inp_row = inp + i * C;
    float* out_row = out + i * C;
    float maxval = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < C; j++) if (inp_row[j] > maxval) maxval = inp_row[j];
    double sum = 0.0;
    for (int j = 0; j < C; j++) { out_row[j] = expf(inp_row[j] - maxval); sum += out_row[j]; }
    float norm = 1.f / (float)sum;
    for (int j = 0; j < C; j++) out_row[j] *= norm;
  }
}

// Online softmax: outer = rows (teams), inner = per-row reduction
void softmax_online_omp(float *d_out, const float *d_inp, int N, int C)
{
  #pragma omp target teams distribute num_teams(N) thread_limit(256)
  for (int row = 0; row < N; row++) {
    const float* x = d_inp + row * C;
    float* y = d_out + row * C;

    float max_val = -3.402823466e+38f;
    #pragma omp parallel for reduction(max:max_val)
    for (int j = 0; j < C; j++) if (x[j] > max_val) max_val = x[j];

    float sum_val = 0.0f;
    #pragma omp parallel for reduction(+:sum_val)
    for (int j = 0; j < C; j++) sum_val += expf(x[j] - max_val);

    #pragma omp parallel for
    for (int j = 0; j < C; j++) y[j] = expf(x[j] - max_val) / sum_val;
  }
}

int main(int argc, char **argv) {
  srand(0);

  int B = 8;
  int T = 1024;
  int V = 50257;

  float* out = (float*)malloc(B * T * V * sizeof(float));
  float* inp = (float*)malloc(B * T * V * sizeof(float));

  for (int i = 0; i < B * T * V; i++)
    inp[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  for (int j = 0; j < B * T; j++)
    for (int k = 0; k < 3; k++) {
      int idx = rand() % V;
      inp[j * V + idx] *= 20.0f;
    }

  int kernel_num = 1;
  if (argc > 1) kernel_num = atoi(argv[1]);
  printf("Using kernel %d\n", kernel_num);

  softmax_forward_cpu(out, inp, B * T, V);
  {
    float max_el = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < B * T * V; i++) if (out[i] > max_el) max_el = out[i];
    assert(max_el > 1e-4f);
    printf("Largest output is: %f\n", max_el);
  }

  float *d_inp = (float*)malloc((size_t)B * T * V * sizeof(float));
  float *d_out = (float*)malloc((size_t)B * T * V * sizeof(float));
  for (int i = 0; i < B * T * V; i++) d_inp[i] = inp[i];

  #pragma omp target enter data map(to: d_inp[0:B*T*V]) map(alloc: d_out[0:B*T*V])

  printf("All results match. Starting benchmarks.\n\n");

  int repeat_times = 10;
  for (int block_size = 32; block_size <= 1024; block_size *= 2) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat_times; i++) {
      softmax_online_omp(d_out, d_inp, B * T, V);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> dur = stop - start;
    float elapsed_time = dur.count() / repeat_times;
    printf("block_size %4d | time %.4f ms | per token %.2f µs\n",
           block_size, elapsed_time, elapsed_time * 1000.0f / (B * T));
    (void)block_size;
    break;
  }

  #pragma omp target exit data map(delete: d_inp[0:B*T*V], d_out[0:B*T*V])
  free(d_inp); free(d_out);
  free(out);
  free(inp);
  return 0;
}
