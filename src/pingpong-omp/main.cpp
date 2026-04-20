#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
  const int    loop_count = 50;
  const int    warmup     = 5;

  printf("%-30s %20s %20s\n",
         "Transfer size (B)", "Transfer Time (s)", "Bandwidth (GB/s)");

  for (int p = 16; p <= 27; ++p) {
    const long int N      = 1L << p;
    const long int num_B  = N * (long int)sizeof(double);
    const double   num_GB = (double)num_B / 1.0e9;

    double *d_src = (double *)malloc((size_t)N * sizeof(double));
    double *d_dst = (double *)malloc((size_t)N * sizeof(double));
    for (long int i = 0; i < N; i++) { d_src[i] = 0.0; d_dst[i] = 0.0; }
#pragma omp target enter data map(to: d_src[0:N], d_dst[0:N])

    // Warmup passes.
    for (int w = 0; w < warmup; ++w) {
#pragma omp target teams distribute parallel for thread_limit(256)
      for (long int i = 0; i < N; i++) d_dst[i] = d_src[i];
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int j = 0; j < loop_count; ++j) {
#pragma omp target teams distribute parallel for thread_limit(256)
      for (long int i = 0; i < N; i++) d_dst[i] = d_src[i];
    }
    auto t1 = std::chrono::steady_clock::now();

    double elapsed  = std::chrono::duration<double>(t1 - t0).count();
    double avg_time = elapsed / (double)loop_count;
    printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
           num_B, avg_time, num_GB / avg_time);

#pragma omp target exit data map(delete: d_src[0:N], d_dst[0:N])
    free(d_src);
    free(d_dst);
  }
  return 0;
}
