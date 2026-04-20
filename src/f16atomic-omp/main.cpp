// OpenMP target offloading port of f16atomic-kokkos
// FP16/BF16 atomicAdd replaced with float32 atomic add.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

void atomicCost(int nelems, int repeat)
{
  const int atomics_count = 256;
  float* d_result = (float*)malloc(nelems * sizeof(float));
  for (int i = 0; i < nelems; i++) d_result[i] = 0.f;

  const float val0 = 0.f;
  const float val1 = 6.1035e-5f;

  #pragma omp target enter data map(tofrom: d_result[0:nelems])

  // warm-up
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int tid = 0; tid < nelems / 2; tid++) {
    float v0 = val0, v1 = val1;
    for (int i = 0; i < atomics_count; i++) {
      #pragma omp atomic update
      d_result[tid * 2] += v0;
      #pragma omp atomic update
      d_result[tid * 2 + 1] += v1;
    }
  }

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int tid = 0; tid < nelems / 2; tid++) {
      float v0 = val0, v1 = val1;
      for (int i = 0; i < atomics_count; i++) {
        #pragma omp atomic update
        d_result[tid * 2] += v0;
        #pragma omp atomic update
        d_result[tid * 2 + 1] += v1;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of float atomic add on global memory: %f (us)\n",
         time * 1e-3f / repeat);

  #pragma omp target update from(d_result[0:nelems])
  printf("First two elements: %f %f\n\n", d_result[0], d_result[1]);

  #pragma omp target exit data map(delete: d_result[0:nelems])
  free(d_result);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  if (nelems <= 0 || nelems % 2 != 0) {
    printf("N must be positive and a multiple of 2\n");
    return 1;
  }

  printf("\nFP32 atomic add (proxy for FP16/BF16)\n");
  atomicCost(nelems, repeat);
  return 0;
}
