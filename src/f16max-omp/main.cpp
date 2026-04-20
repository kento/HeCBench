// OpenMP target offloading port of f16max-kokkos
// FP16 half2 max replaced with float fmaxf.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define NUM_OF_BLOCKS  1048576
#define NUM_OF_THREADS 256

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;

  float* a = (float*)malloc(size * sizeof(float));
  float* b = (float*)malloc(size * sizeof(float));
  float* r = (float*)malloc(size * sizeof(float));

  srand(123);
  for (size_t i = 0; i < size; i++) {
    a[i] = (float)(rand() % 922021);
    b[i] = (float)(rand() % 922021);
  }

  #pragma omp target enter data map(to: a[0:size], b[0:size]) \
                                map(alloc: r[0:size])

  // warm-up
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (size_t i = 0; i < size; i++) {
    r[i] = fmaxf(a[i], b[i]);
  }

  auto start = std::chrono::steady_clock::now();
  for (int rep = 0; rep < repeat; rep++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < size; i++) {
      r[i] = fmaxf(a[i], b[i]);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target update from(r[0:size])

  #pragma omp target exit data map(delete: a[0:size], b[0:size], r[0:size])

  bool ok = true;
  for (size_t i = 0; i < size; i++) {
    float expected = fmaxf(a[i], b[i]);
    if (fabsf(r[i] - expected) > 1e-3f) { ok = false; break; }
  }
  printf("fmax result: %s\n", ok ? "PASS" : "FAIL");

  free(a); free(b); free(r);
  return ok ? 0 : 1;
}
