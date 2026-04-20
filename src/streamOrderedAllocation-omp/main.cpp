// OpenMP target offloading port of streamOrderedAllocation benchmark
// Sequential malloc + target map allocation pattern

#include <omp.h>
#include <stdio.h>
#include <climits>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#define repeat 100

static int basicAllocation(const int nelem, const float *a,
                           const float *b, float *c, bool timing) {
  float *d_a, *d_b, *d_c;
  size_t bytes = nelem * sizeof(float);

  printf("Starting basicAllocation()\n");

  d_a = (float *)malloc(bytes);
  d_b = (float *)malloc(bytes);
  d_c = (float *)malloc(bytes);

  memcpy(d_a, a, bytes);
  memcpy(d_b, b, bytes);
  memset(d_c, 0, bytes);

  #pragma omp target enter data map(to: d_a[0:nelem], d_b[0:nelem]) \
                                  map(alloc: d_c[0:nelem])

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < nelem; idx++)
      d_c[idx] = d_a[idx] + d_b[idx];
  }

  auto end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
  if (timing) printf("  Time: %.6f s\n", elapsed);

  #pragma omp target update from(d_c[0:nelem])
  #pragma omp target exit data map(delete: d_a[0:nelem], d_b[0:nelem], d_c[0:nelem])

  // Verify
  float errorNorm = 0.f, refNorm = 0.f;
  for (int i = 0; i < nelem; i++) {
    float ref  = a[i] + b[i];
    float diff = d_c[i] - ref;
    errorNorm += diff * diff;
    refNorm   += ref  * ref;
  }
  errorNorm = sqrtf(errorNorm);
  refNorm   = sqrtf(refNorm);

  free(d_a); free(d_b); free(d_c);

  if (errorNorm / refNorm > 1e-6f) { printf("basicAllocation FAILED\n"); return 1; }
  printf("basicAllocation PASSED\n");
  return 0;
}

// Post-sync version: simulates pool-based allocation where sync doesn't free memory
static int streamOrderedAllocationPostSync(const int nelem, const float *a,
                                           const float *b, float *c) {
  size_t bytes = nelem * sizeof(float);
  printf("Starting streamOrderedAllocationPostSync()\n");

  float *d_a = (float *)malloc(bytes);
  float *d_b = (float *)malloc(bytes);
  float *d_c = (float *)malloc(bytes);

  memcpy(d_a, a, bytes);
  memcpy(d_b, b, bytes);
  memset(d_c, 0, bytes);

  // First allocation + compute
  #pragma omp target data map(to: d_a[0:nelem], d_b[0:nelem]) map(tofrom: d_c[0:nelem])
  {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < nelem; idx++)
      d_c[idx] = d_a[idx] + d_b[idx];
  }

  // Synchronize (data is still logically "pooled" — no actual free)
  memcpy(c, d_c, bytes);

  // Second allocation simulating re-use from pool
  float *d_a2 = (float *)malloc(bytes);
  float *d_b2 = (float *)malloc(bytes);
  float *d_c2 = (float *)malloc(bytes);
  memcpy(d_a2, a, bytes);
  memcpy(d_b2, b, bytes);
  memset(d_c2, 0, bytes);

  #pragma omp target data map(to: d_a2[0:nelem], d_b2[0:nelem]) map(tofrom: d_c2[0:nelem])
  {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < nelem; idx++)
      d_c2[idx] = d_a2[idx] + d_b2[idx];
  }

  free(d_a); free(d_b); free(d_c);
  free(d_a2); free(d_b2); free(d_c2);

  // Verify
  float errorNorm = 0.f, refNorm = 0.f;
  for (int i = 0; i < nelem; i++) {
    float ref  = a[i] + b[i];
    float diff = c[i] - ref;
    errorNorm += diff * diff;
    refNorm   += ref  * ref;
  }
  errorNorm = sqrtf(errorNorm);
  refNorm   = sqrtf(refNorm);

  if (errorNorm / refNorm > 1e-6f) { printf("streamOrderedAllocationPostSync FAILED\n"); return 1; }
  printf("streamOrderedAllocationPostSync PASSED\n");
  return 0;
}

int main(int argc, char **argv) {
  const int nelem = 1 << 20; // 1M elements

  std::vector<float> a(nelem), b(nelem), c(nelem, 0.f);
  srand(42);
  for (int i = 0; i < nelem; i++) {
    a[i] = rand() / (float)RAND_MAX;
    b[i] = rand() / (float)RAND_MAX;
  }

  int rc = 0;
  rc += basicAllocation(nelem, a.data(), b.data(), c.data(), true);
  rc += streamOrderedAllocationPostSync(nelem, a.data(), b.data(), c.data());

  return rc;
}
