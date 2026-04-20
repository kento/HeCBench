// OpenMP target offloading port of rotary benchmark.
// Rotary position encoding: o1 = x1*cos - x2*sin, o2 = x1*sin + x2*cos.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  constexpr int num_threads_val  = 32 * 4;
  constexpr int thread_work_size = 4;
  constexpr int block_work_size  = thread_work_size * num_threads_val;
  const long numel = (long)block_work_size * 10000;

  printf("Number of elements: %ld\n", numel);

  float* h_x1  = (float*)malloc(numel * sizeof(float));
  float* h_x2  = (float*)malloc(numel * sizeof(float));
  float* h_cos = (float*)malloc(numel * sizeof(float));
  float* h_sin = (float*)malloc(numel * sizeof(float));
  float* h_o1  = (float*)malloc(numel * sizeof(float));
  float* h_o2  = (float*)malloc(numel * sizeof(float));

  for (long i = 0; i < numel; i++) {
    h_x1[i]  = (float)(i + 1) / numel;
    h_x2[i]  = (float)(i + 1) / numel;
    h_cos[i] = cosf((float)i / powf(10000.f, (float)i / numel));
    h_sin[i] = sinf((float)i / powf(10000.f, (float)i / numel));
  }

  float* d_x1  = new float[numel];
  float* d_x2  = new float[numel];
  float* d_cos = new float[numel];
  float* d_sin = new float[numel];
  float* d_o1  = new float[numel];
  float* d_o2  = new float[numel];

  for (long i = 0; i < numel; i++) {
    d_x1[i] = h_x1[i]; d_x2[i] = h_x2[i];
    d_cos[i] = h_cos[i]; d_sin[i] = h_sin[i];
  }

#pragma omp target enter data \
  map(to: d_x1[0:numel], d_x2[0:numel], d_cos[0:numel], d_sin[0:numel]) \
  map(alloc: d_o1[0:numel], d_o2[0:numel])

  // Warmup
#pragma omp target teams distribute parallel for thread_limit(256)
  for (long i = 0; i < numel; i++) {
    d_o1[i] = d_x1[i] * d_cos[i] - d_x2[i] * d_sin[i];
    d_o2[i] = d_x1[i] * d_sin[i] + d_x2[i] * d_cos[i];
  }

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (long i = 0; i < numel; i++) {
      d_o1[i] = d_x1[i] * d_cos[i] - d_x2[i] * d_sin[i];
      d_o2[i] = d_x1[i] * d_sin[i] + d_x2[i] * d_cos[i];
    }
  }
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (float)time * 1e-3f / repeat);

#pragma omp target update from(d_o1[0:numel], d_o2[0:numel])
  for (long i = 0; i < numel; i++) { h_o1[i] = d_o1[i]; h_o2[i] = d_o2[i]; }

  bool ok = true;
  for (long i = 0; i < numel; i++) {
    float r1 = h_x1[i] * h_cos[i] - h_x2[i] * h_sin[i];
    float r2 = h_x1[i] * h_sin[i] + h_x2[i] * h_cos[i];
    if (fabsf(r1 - h_o1[i]) > 1e-3f || fabsf(r2 - h_o2[i]) > 1e-3f) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

#pragma omp target exit data \
  map(delete: d_x1[0:numel], d_x2[0:numel], d_cos[0:numel], d_sin[0:numel], \
              d_o1[0:numel], d_o2[0:numel])

  free(h_x1); free(h_x2); free(h_cos); free(h_sin); free(h_o1); free(h_o2);
  delete[] d_x1; delete[] d_x2; delete[] d_cos; delete[] d_sin; delete[] d_o1; delete[] d_o2;
  return 0;
}
