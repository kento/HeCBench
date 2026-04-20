// NCCL/MPI AllReduce bandwidth benchmark – OpenMP target offloading port
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int nRanks = 1;

  for (long size = 1024LL * 1024LL;
       size <= 1000LL * 1024LL * 1024LL;
       size *= 10) {

    float* d_buf = (float*)malloc(size * sizeof(float));
    #pragma omp target enter data map(alloc: d_buf[0:size])

    // Fill with 1.0f
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (long i = 0; i < size; i++) d_buf[i] = 1.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < repeat; ++r) {
      float sum = 0.0f;
      #pragma omp target teams distribute parallel for reduction(+:sum) thread_limit(256)
      for (long i = 0; i < size; i++) sum += d_buf[i];

      const float bcast_val = (float)nRanks;
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (long i = 0; i < size; i++) d_buf[i] = bcast_val;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Verify
    float h_val;
    #pragma omp target update from(d_buf[0:1])
    h_val = d_buf[0];
    bool ok = (h_val == (float)nRanks);

    long   num_B  = (long)sizeof(float) * size * nRanks;
    double num_GB = (double)num_B / (double)(1LL << 30);
    double avg_t  = elapsed / repeat;

    printf("Transfer size (B): %10li, Average Transfer Time (s): %15.9f, "
           "Bandwidth (GB/s): %15.9f\n",
           num_B, avg_t, num_GB / avg_t);
    printf("%s\n", ok ? "PASS" : "FAIL");

    #pragma omp target exit data map(delete: d_buf[0:size])
    free(d_buf);
  }
  return 0;
}
