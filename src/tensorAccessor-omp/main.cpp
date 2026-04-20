// OpenMP target offloading port of tensorAccessor benchmark
// Matrix-vector multiply with raw and accessor-style indexing

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <nrows> <ncols> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t nrow   = atol(argv[1]);
  const int64_t ncol   = atol(argv[2]);
  const int      repeat = atoi(argv[3]);

  const int64_t numel = nrow * ncol;

  std::vector<float> h_m(numel), h_v(ncol), h_r(nrow), h_r_ref(nrow);
  srand(123);
  for (int64_t i = 0; i < numel; i++) h_m[i] = rand() / (float)RAND_MAX;
  for (int64_t i = 0; i < ncol;  i++) h_v[i] = rand() / (float)RAND_MAX;

  // Reference on CPU
  for (int64_t i = 0; i < nrow; i++) {
    float val = 0.f;
    for (int64_t j = 0; j < ncol; j++) val += h_m[i * ncol + j] * h_v[j];
    h_r_ref[i] = val;
  }

  float *d_m = h_m.data();
  float *d_v = h_v.data();
  float *d_r = h_r.data();

  #pragma omp target enter data map(to: d_m[0:numel], d_v[0:ncol]) \
                                  map(alloc: d_r[0:nrow])

  // Warmup
  printf("Warmup..\n");
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < nrow; row++) {
      float val = 0.f;
      for (int64_t j = 0; j < ncol; j++) val += d_m[row * ncol + j] * d_v[j];
      d_r[row] = val;
    }
  }

  // Benchmark raw accessor kernel
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < nrow; row++) {
      float val = 0.f;
      for (int64_t j = 0; j < ncol; j++) val += d_m[row * ncol + j] * d_v[j];
      d_r[row] = val;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  printf("Average execution time of raw_accessor_kernel: %f (us)\n", ns * 1e-3f / repeat);

  // Benchmark tensor accessor kernel (same computation)
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < nrow; row++) {
      float val = 0.f;
      for (int64_t j = 0; j < ncol; j++) val += d_m[row * ncol + j] * d_v[j];
      d_r[row] = val;
    }
  }
  t1 = std::chrono::steady_clock::now();
  ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  printf("Average execution time of tensor_packed_accessor_kernel: %f (us)\n", ns * 1e-3f / repeat);

  #pragma omp target update from(d_r[0:nrow])
  #pragma omp target exit data map(delete: d_m[0:numel], d_v[0:ncol], d_r[0:nrow])

  bool ok = true;
  for (int64_t i = 0; i < nrow; i++) {
    if (fabsf(d_r[i] - h_r_ref[i]) > 1e-3f) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
