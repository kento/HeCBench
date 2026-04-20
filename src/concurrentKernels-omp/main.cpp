// concurrentKernels benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr long long CLOCK_COUNT = 1000000LL;

int main(int argc, char* argv[]) {
  int nkernels = (argc > 1) ? std::atoi(argv[1]) : 4;
  if (nkernels < 1) { fprintf(stderr, "nkernels must be >= 1\n"); return 1; }

  long long* d_results = (long long*)malloc(nkernels * sizeof(long long));
  memset(d_results, 0, nkernels * sizeof(long long));

  #pragma omp target enter data map(alloc: d_results[0:nkernels])
  #pragma omp target update to(d_results[0:nkernels])

  for (int k = 0; k < nkernels; ++k) {
    long long kernel_sum = 0;
    #pragma omp target teams distribute parallel for reduction(+:kernel_sum) thread_limit(256)
    for (long long j = 0; j < CLOCK_COUNT; j++) {
      kernel_sum += j % 3;
    }
    d_results[k] = kernel_sum;
    #pragma omp target update to(d_results[k:1])
  }

  long long total = 0;
  #pragma omp target teams distribute parallel for reduction(+:total) thread_limit(256)
  for (int i = 0; i < nkernels; i++) {
    total += d_results[i];
  }

  long long full_cycles = CLOCK_COUNT / 3;
  long long rem = CLOCK_COUNT % 3;
  long long single_sum = full_cycles * 3;
  for (long long r = 0; r < rem; ++r) single_sum += r;
  long long expected = (long long)nkernels * single_sum;

  printf("nkernels    : %d\n", nkernels);
  printf("clock_count : %lld\n", CLOCK_COUNT);
  printf("single_sum  : %lld\n", single_sum);
  printf("total       : %lld\n", total);
  printf("expected    : %lld\n", expected);
  printf("%s\n", (total == expected) ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: d_results[0:nkernels])
  free(d_results);
  return 0;
}
