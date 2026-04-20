// Dropout benchmark – OpenMP target offloading port
// LCG-based per-element random number; curand replaced.
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <omp.h>

#pragma omp declare target
float lcg_rand(uint64_t& state)
{
  state = 6364136223846793005ULL * state + 1442695040888963407ULL;
  return (float)((state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nelem  = atol(argv[1]);
  const int      repeat = atoi(argv[2]);

  float*   d_a    = (float*)  malloc(nelem * sizeof(float));
  float*   d_b    = (float*)  malloc(nelem * sizeof(float));
  uint8_t* d_mask = (uint8_t*)malloc(nelem * sizeof(uint8_t));

  #pragma omp target enter data map(alloc: d_a[0:nelem], d_b[0:nelem], d_mask[0:nelem])

  // Initialise input
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int64_t i = 0; i < nelem; i++)
    d_a[i] = 0.1f;

  auto start = std::chrono::steady_clock::now();

  for (int p = 1; p <= repeat; p++) {
    float pa    = (float)p / repeat;
    float scale = 1.f / pa;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t i = 0; i < nelem; i++) {
      uint64_t s = (uint64_t)(12345678ULL * (i + 1) + 87654321ULL * p);
      float r = lcg_rand(s);
      uint8_t keep = (r < pa) ? 1 : 0;
      d_b[i]    = d_a[i] * keep * scale;
      d_mask[i] = keep;
    }
  }

  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time %lf (s)\n", time * 1e-9);

  #pragma omp target exit data map(delete: d_a[0:nelem], d_b[0:nelem], d_mask[0:nelem])

  free(d_a);
  free(d_b);
  free(d_mask);
  return 0;
}
