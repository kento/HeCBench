// cmembench – OpenMP target offloading port
// Benchmarks read-only memory bandwidth using scalar, 2-wide, and 4-wide access.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define VECTOR_SIZE 1024

static constexpr long long TOTAL_INTS = 4096LL * VECTOR_SIZE;

double run_bench(const int* data, int stride, int repeat) {
  const long long n = TOTAL_INTS / stride;
  const int vs = VECTOR_SIZE;

  // Warm-up
  {
    long long sum = 0;
    #pragma omp target teams distribute parallel for reduction(+:sum) thread_limit(256)
    for (long long i = 0; i < n; i++) {
      long long s = 0;
      for (int k = 0; k < stride; ++k)
        s += data[(i * stride + k) % vs];
      sum += s;
    }
  }

  long long verified_sum = -1;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; ++r) {
    long long sum = 0;
    #pragma omp target teams distribute parallel for reduction(+:sum) thread_limit(256)
    for (long long i = 0; i < n; i++) {
      long long s = 0;
      for (int k = 0; k < stride; ++k)
        s += data[(i * stride + k) % vs];
      sum += s;
    }
    if (r == 0) verified_sum = sum;
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  if (verified_sum == TOTAL_INTS)
    printf("  Verification PASS (sum=%lld)\n", verified_sum);
  else
    printf("  Verification FAIL: sum=%lld expected=%lld\n", verified_sum, TOTAL_INTS);

  double elapsed_s =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9;
  double bytes = (double)TOTAL_INTS * sizeof(int) * repeat;
  return bytes / elapsed_s / 1e9;
}

int main(int argc, char* argv[]) {
  int repeat = (argc > 1) ? std::atoi(argv[1]) : 100;

  int* d_data = (int*)malloc(VECTOR_SIZE * sizeof(int));
  for (int i = 0; i < VECTOR_SIZE; ++i) d_data[i] = 1;

  #pragma omp target enter data map(alloc: d_data[0:VECTOR_SIZE])
  #pragma omp target update to(d_data[0:VECTOR_SIZE])

  printf("cmembench: VECTOR_SIZE=%d, TOTAL_INTS=%lld, repeat=%d\n",
         VECTOR_SIZE, TOTAL_INTS, repeat);
  printf("%-12s  %10s\n", "Access", "BW (GB/s)");

  double bw;
  bw = run_bench(d_data, 1, repeat);
  printf("%-12s  %10.3f\n", "int (x1)", bw);

  bw = run_bench(d_data, 2, repeat);
  printf("%-12s  %10.3f\n", "int2 (x2)", bw);

  bw = run_bench(d_data, 4, repeat);
  printf("%-12s  %10.3f\n", "int4 (x4)", bw);

  #pragma omp target exit data map(delete: d_data[0:VECTOR_SIZE])
  free(d_data);
  return 0;
}
