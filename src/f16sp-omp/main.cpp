// OpenMP target offloading port of f16sp-kokkos (FP16 scalar/dot product benchmark)
// half2 replaced with float pairs; cuBLAS dot replaced with OMP reduction.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define NUM_OF_BLOCKS  (1024 * 1024)
#define NUM_OF_THREADS 128

void generateInput(float* ax, float* ay, size_t size)
{
  float v = sqrtf(32752.f / size);
  for (size_t i = 0; i < size; i++) { ax[i] = v; ay[i] = v; }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;

  float* ax = (float*)malloc(size * sizeof(float));
  float* ay = (float*)malloc(size * sizeof(float));
  float* bx = (float*)malloc(size * sizeof(float));
  float* by = (float*)malloc(size * sizeof(float));

  srand(123);
  generateInput(ax, ay, size);
  generateInput(bx, by, size);

  printf("\nNumber of elements in the vectors is %zu\n", size * 2);

  #pragma omp target enter data map(to: ax[0:size], ay[0:size], bx[0:size], by[0:size])

  for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid /= 2) {
    printf("\nGrid size is %d\n", grid);

    // warm-up
    for (int i = 0; i < 10; i++) {
      float r = 0.f;
      #pragma omp target teams distribute parallel for reduction(+:r) thread_limit(256)
      for (size_t j = 0; j < size; j++) {
        r += ax[j] * bx[j] + ay[j] * by[j];
      }
      (void)r;
    }

    float r = 0.f;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      r = 0.f;
      #pragma omp target teams distribute parallel for reduction(+:r) thread_limit(256)
      for (size_t j = 0; j < size; j++) {
        r += ax[j] * bx[j] + ay[j] * by[j];
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel1 execution time %f (us)\n", (t * 1e-3f) / repeat);
    printf("Error rate: %e\n", fabsf(r - 65504.f) / 65504.f);
  }

  #pragma omp target exit data map(delete: ax[0:size], ay[0:size], bx[0:size], by[0:size])

  free(ax); free(ay); free(bx); free(by);
  return 0;
}
